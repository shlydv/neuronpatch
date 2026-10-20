"""Experimental per-occurrence output-vector edits, matching FiNE's parameter layout."""

import json
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .grouped import GroupedRowPatch
from .layers import ColumnPatch, RowPatch
from .patch import fingerprint

_DTYPES = {name: getattr(torch, name) for name in ("float16", "bfloat16", "float32", "float64")}


class OccurrenceColumnEdit(nn.Module):
    """One global delta row per selected (module name, input column) occurrence.

    Repeated columns retain independent trainable rows. The global [occurrences,
    output width] layout permits the same joint norm and Adam state as FiNE.
    All target projections must have equal output width and share one device.
    delta_dtype can differ from base weights, as in FP16-base/BF16-delta FiNE.

    save()/load() store compact deltas against an exact base fingerprint.
    Ordinary state_dict also includes the frozen base. Use applied() around
    model.generate(), or call this
    module for a forward pass. Contexts remove their hooks even on exceptions.
    Forward hooks are scoped but not thread-safe; do not concurrently edit or
    mutate the same model. Numerical equivalence to merged weights is subject to
    mixed-precision and accumulation rounding, as in the reference editor.
    """

    def __init__(
        self,
        model: nn.Module,
        targets: Sequence[tuple[str, int]],
        *,
        delta_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        values = tuple(targets)
        if not values or any(
            isinstance(module, (ColumnPatch, RowPatch, GroupedRowPatch, OccurrenceColumnEdit))
            for module in model.modules()
        ):
            raise ValueError("provide occurrence targets on an unpatched model")
        modules = dict(model.named_modules(remove_duplicate=False))
        parameters = list(model.named_parameters(remove_duplicate=False))
        validated = {}
        assignments = {}
        for occurrence, entry in enumerate(values):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("each target must contain a module name and column")
            name, column = entry
            module = modules.get(name) if isinstance(name, str) else None
            if not name or type(module) is not nn.Linear:
                raise ValueError("targets must name ordinary nn.Linear submodules")
            if type(column) is not int or not 0 <= column < module.in_features:
                raise ValueError("column outside target projection")
            if module.weight.device.type == "meta" or not module.weight.is_floating_point():
                raise ValueError("targets must be materialized floating-point projections")
            if sum(m is module for m in modules.values()) != 1 or any(
                sum(p is parameter for _, p in parameters) != 1 for parameter in module.parameters()
            ):
                raise ValueError("shared target modules or parameters are unsupported")
            validated[name] = module
            assignments.setdefault(name, []).append((column, occurrence))
        widths = {module.out_features for module in validated.values()}
        devices = {module.weight.device for module in validated.values()}
        if len(widths) != 1 or len(devices) != 1:
            raise ValueError("target projections must share output width and device")
        dtype = delta_dtype or next(iter(validated.values())).weight.dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("delta_dtype must be a supported floating-point dtype")
        # Validate every target before freezing any original parameters.
        self.base_fingerprint = fingerprint(model)
        self.model = model
        self.model.requires_grad_(False)
        self.targets = tuple((name, column) for name, column in values)
        device = next(iter(devices))
        self.delta = nn.Parameter(
            torch.zeros(len(values), next(iter(widths)), device=device, dtype=dtype)
        )
        self._groups = []
        for group, (name, entries) in enumerate(assignments.items()):
            columns, occurrences = zip(*entries, strict=True)
            self.register_buffer(f"columns_{group}", torch.tensor(columns, device=device))
            self.register_buffer(f"occurrences_{group}", torch.tensor(occurrences, device=device))
            self._groups.append((name, group))
        self._active = False

    def trainable_parameters(self):
        yield self.delta

    @property
    def num_trainable_parameters(self):
        return self.delta.numel()

    def delta_norm(self):
        """Joint norm over occurrences, rather than a sum of per-layer norms."""
        return torch.norm(self.delta, p=2)

    @contextmanager
    def applied(self):
        if self._active:
            raise RuntimeError("edit context is already active")
        self._active = True
        handles = []
        try:
            for name, group in self._groups:

                def hook(module, inputs, output, group=group):
                    features = inputs[0].index_select(-1, getattr(self, f"columns_{group}"))
                    deltas = self.delta.index_select(0, getattr(self, f"occurrences_{group}"))
                    update = features.to(self.delta.dtype) @ deltas
                    return output + update.to(inputs[0].dtype)

                handles.append(self.model.get_submodule(name).register_forward_hook(hook))
            yield self.model
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    def forward(self, *args, **kwargs):
        with self.applied():
            return self.model(*args, **kwargs)

    def save(self, directory: str | Path, *, metadata: dict | None = None):
        """Save only ordered occurrences and their deltas, not base/optimizer state."""
        if self._active:
            raise RuntimeError("exit the edit context before saving")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON-serializable dictionary")
        if not torch.isfinite(self.delta).all():
            raise ValueError("cannot save nonfinite deltas")
        if fingerprint(self.model) != self.base_fingerprint:
            raise ValueError("base model changed since edit creation")
        config = {
            "format_version": 1,
            "method": "occurrence_columns",
            "base_fingerprint": self.base_fingerprint,
            "targets": self.targets,
            "delta_dtype": str(self.delta.dtype).removeprefix("torch."),
            "metadata": metadata or {},
        }
        serialized = json.dumps(config, indent=2) + "\n"
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        save_file(
            {"delta": self.delta.detach().cpu().contiguous()}, str(destination / "edit.safetensors")
        )
        (destination / "edit.json").write_text(serialized)

    @classmethod
    def load(cls, model: nn.Module, directory: str | Path):
        """Validate an occurrence artifact before freezing or modifying the base."""
        source = Path(directory)
        config = json.loads((source / "edit.json").read_text())
        if (
            not isinstance(config, dict)
            or type(config.get("format_version")) is not int
            or config["format_version"] != 1
            or config.get("method") != "occurrence_columns"
        ):
            raise ValueError("unsupported occurrence edit format")
        dtype_name = config.get("delta_dtype")
        dtype = _DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        targets = config.get("targets")
        if dtype is None or not isinstance(targets, list) or not targets:
            raise ValueError("invalid occurrence dtype or targets")
        tensors = load_file(str(source / "edit.safetensors"))
        if set(tensors) != {"delta"}:
            raise ValueError("occurrence artifact must contain only delta")
        delta = tensors["delta"]
        if (
            delta.ndim != 2
            or delta.shape[0] != len(targets)
            or delta.dtype != dtype
            or not torch.isfinite(delta).all()
        ):
            raise ValueError("invalid occurrence delta shape, dtype or values")
        for entry in targets:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or type(entry[1]) is not int
            ):
                raise ValueError("invalid occurrence target")
            try:
                module = model.get_submodule(entry[0])
            except AttributeError as error:
                raise ValueError("unknown occurrence target") from error
            if type(module) is not nn.Linear or delta.shape[1] != module.out_features:
                raise ValueError("occurrence target shape mismatch")
        if fingerprint(model) != config.get("base_fingerprint"):
            raise ValueError("base checkpoint mismatch; use original weights and dtype")
        edit = cls(model, targets, delta_dtype=dtype)
        with torch.no_grad():
            edit.delta.copy_(delta.to(edit.delta.device))
        return edit

    @torch.no_grad()
    def merged_model(self):
        """Copy the base and apply occurrences sequentially, preserving update order.

        Requires memory for a second model. Do not expect bit-identical logits
        between live mixed-precision hooks and the merged model.
        """
        if self._active:
            raise RuntimeError("exit the edit context before merging")
        result = deepcopy(self.model)
        for occurrence, (name, column) in enumerate(self.targets):
            weight = result.get_submodule(name).weight
            weight[:, column].add_(self.delta[occurrence].to(weight.device))
        return result
