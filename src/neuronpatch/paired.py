"""Experimental paired incoming-row/outgoing-column MLP updates.

This is an explicit parameterization, not a neuron selector or a validated
adaptation recipe. A shared intermediate index connects the supplied projections;
the caller must verify their actual computation graph. No concept/circuit claim
follows from shape compatibility. Ordinary and gated MLPs are supported.
"""

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from .grouped import GroupedRowPatch
from .layers import ColumnPatch, RowPatch, validate_indices
from .patch import _fingerprint_state, _replace_submodule, fingerprint

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class MLPSelection:
    """Explicit intermediate indices and ordinary Linear projection paths.

    For a gated MLP, incoming contains both gate and up projections. Their
    selected output rows and the outgoing projection's matching input columns
    receive independent deltas. Biases remain frozen.
    """

    incoming: tuple[str, ...]
    outgoing: str
    indices: tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.incoming, str):
            raise ValueError("incoming must be a sequence of projection paths")
        object.__setattr__(self, "incoming", tuple(self.incoming))
        object.__setattr__(self, "indices", tuple(self.indices))


class _PreciseRowPatch(RowPatch):
    def forward(self, x):
        output = self.base(x)
        # Keep small trainables/updates in their explicit dtype under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(x.to(self.delta.dtype), self.delta)
        return output.index_add(-1, self.indices, update.to(output.dtype))

    @torch.no_grad()
    def merged_linear(self):
        merged = deepcopy(self.base)
        dtype = torch.promote_types(merged.weight.dtype, self.delta.dtype)
        selected = merged.weight.index_select(0, self.indices).to(dtype)
        merged.weight.index_copy_(0, self.indices, (selected + self.delta).to(merged.weight.dtype))
        return merged


class _PreciseColumnPatch(ColumnPatch):
    def forward(self, x):
        output = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(x.index_select(-1, self.indices).to(self.delta.dtype), self.delta)
        return output + update.to(output.dtype)

    @torch.no_grad()
    def merged_linear(self):
        merged = deepcopy(self.base)
        dtype = torch.promote_types(merged.weight.dtype, self.delta.dtype)
        selected = merged.weight.index_select(1, self.indices).to(dtype)
        merged.weight.index_copy_(1, self.indices, (selected + self.delta).to(merged.weight.dtype))
        return merged


def _layout(model, selections):
    if not selections or any(not isinstance(s, MLPSelection) for s in selections):
        raise ValueError("provide a nonempty sequence of MLPSelection objects")
    if any(isinstance(m, (ColumnPatch, RowPatch, GroupedRowPatch)) for m in model.modules()):
        raise ValueError("provide an unpatched model")
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = list(model.named_parameters(remove_duplicate=False))
    result = {}
    for selection in selections:
        paths = (*selection.incoming, selection.outgoing)
        if not selection.incoming or any(not isinstance(n, str) or not n for n in paths):
            raise ValueError("nonempty incoming and outgoing projection paths required")
        for name in paths:
            module = modules.get(name)
            if name in result or type(module) is not nn.Linear:
                raise ValueError("projection paths must be unique ordinary nn.Linear targets")
            if module.weight.device.type == "meta" or not module.weight.is_floating_point():
                raise ValueError("targets must be materialized floating-point projections")
            if sum(m is module for m in modules.values()) != 1 or any(
                sum(p is parameter for _, p in parameters) != 1 for parameter in module.parameters()
            ):
                raise ValueError("shared projections or tied target parameters are unsupported")
            axis = "input" if name == selection.outgoing else "output"
            width = module.in_features if axis == "input" else module.out_features
            result[name] = (module, axis, validate_indices(selection.indices, width))
        outgoing = modules[selection.outgoing]
        if any(
            modules[n].out_features != outgoing.in_features
            or modules[n].in_features != outgoing.out_features
            or modules[n].weight.device != outgoing.weight.device
            for n in selection.incoming
        ):
            raise ValueError("MLP projections must share hidden/intermediate widths and device")
    return result


class PairedMLPPatch:
    """Attach paired updates in-place; optimize trainable_parameters().

    All original parameters are frozen. New deltas default to FP32 even on an
    FP16/BF16 base. For hidden width h and k selected intermediate indices, an
    ordinary MLP trains 2kh scalars and a gated MLP trains 3kh, not k scalars.
    Projections are supplied explicitly; no architecture heuristics or hooks.
    train_incoming=False creates an outgoing-only experimental control while
    validating the same MLP connectivity shapes. Its artifact uses format2;
    ordinary paired artifacts retain format1 and load older format1 manifests.

    The exact original base must remain unchanged. Save/load validates its tensor
    fingerprint. Merging uses promoted addition then casts once to base dtype;
    live and merged predictions can differ through floating-point rounding.
    This class is experimental and is intentionally not in the top-level API.
    """

    def __init__(
        self,
        model: nn.Module,
        selections: Sequence[MLPSelection],
        *,
        delta_dtype: torch.dtype = torch.float32,
        train_incoming: bool = True,
    ):
        if delta_dtype not in _DTYPES.values():
            raise ValueError("delta_dtype must be float32 or float64")
        if type(train_incoming) is not bool:
            raise ValueError("train_incoming must be boolean")
        selections = tuple(selections)
        checked = _layout(model, selections)
        if not train_incoming:
            checked = {name: value for name, value in checked.items() if value[1] == "input"}
        self.base_fingerprint = fingerprint(model)
        self.model = model
        self.selections = selections
        self.delta_dtype = delta_dtype
        self.train_incoming = train_incoming
        self._merged = False
        self.layers = {}
        self._indices = {name: indices for name, (_, _, indices) in checked.items()}
        self._shapes = {}
        model.requires_grad_(False)
        for name, (module, axis, indices) in checked.items():
            layer = (_PreciseColumnPatch if axis == "input" else _PreciseRowPatch)(module, indices)
            layer.delta = nn.Parameter(layer.delta.to(delta_dtype))
            layer.train(module.training)
            _replace_submodule(model, name, layer)
            self.layers[name] = layer
            self._shapes[name] = layer.delta.shape

    def trainable_parameters(self):
        return (layer.delta for layer in self.layers.values())

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def _check_base(self):
        if self._merged:
            raise RuntimeError("patch was already merged or unloaded")
        state = self.model.state_dict()
        for name, layer in self.layers.items():
            if self.model.get_submodule(name) is not layer:
                raise ValueError("patched projection was replaced")
            if (
                layer.indices.dtype != torch.long
                or tuple(layer.indices.detach().cpu().tolist()) != self._indices[name]
                or layer.delta.dtype != self.delta_dtype
                or layer.delta.shape != self._shapes[name]
            ):
                raise ValueError("patch indices, trainable shape or dtype changed")
            for suffix in ("delta", "indices"):
                state.pop(f"{name}.{suffix}")
            for suffix in ("weight", "bias"):
                key = f"{name}.base.{suffix}"
                if key in state:
                    state[f"{name}.{suffix}"] = state.pop(key)
        if _fingerprint_state(state) != self.base_fingerprint:
            raise ValueError("base model changed since patch creation")

    @torch.no_grad()
    def merged_model(self):
        """Return a separate ordinary model; requires memory for a second copy."""
        self._check_base()
        if any(not torch.isfinite(p).all() for p in self.trainable_parameters()):
            raise ValueError("cannot merge nonfinite deltas")
        result = deepcopy(self.model)
        for name in self.layers:
            _replace_submodule(result, name, result.get_submodule(name).merged_linear())
        return result

    @torch.no_grad()
    def merge_and_unload(self):
        """Merge in-place without a second model; invalidate this patch owner.

        Precompute/check every selected merged slice before any mutation. This
        avoids a partial merge on nonfinite updates/overflow. The caller must
        save the compact patch first and retain an original checkpoint to undo
        the merge. The base weights are deliberately changed by this operation.
        """
        self._check_base()
        updates = {}
        for name, layer in self.layers.items():
            axis = 1 if isinstance(layer, ColumnPatch) else 0
            weight = layer.base.weight
            dtype = torch.promote_types(weight.dtype, layer.delta.dtype)
            update = (weight.index_select(axis, layer.indices).to(dtype) + layer.delta).to(
                weight.dtype
            )
            if not torch.isfinite(layer.delta).all() or not torch.isfinite(update).all():
                raise ValueError("cannot merge nonfinite deltas or overflowing weights")
            updates[name] = (axis, update)
        for name, layer in self.layers.items():
            axis, update = updates[name]
            layer.base.weight.index_copy_(axis, layer.indices, update)
            _replace_submodule(self.model, name, layer.base)
        self.layers.clear()
        self._merged = True
        return self.model

    def unload(self):
        """Remove live deltas without merging; preserve the original base weights."""
        self._check_base()
        for name, layer in self.layers.items():
            _replace_submodule(self.model, name, layer.base)
        self.layers.clear()
        self._merged = True
        return self.model

    def save(self, directory: str | Path, *, metadata: dict | None = None):
        self._check_base()
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON-serializable dictionary")
        if any(not torch.isfinite(p).all() for p in self.trainable_parameters()):
            raise ValueError("cannot save nonfinite deltas")
        config = {
            "format_version": 1 if self.train_incoming else 2,
            "method": "paired_mlp",
            "train_incoming": self.train_incoming,
            "base_fingerprint": self.base_fingerprint,
            "delta_dtype": str(self.delta_dtype).removeprefix("torch."),
            "selections": [
                {"incoming": s.incoming, "outgoing": s.outgoing, "indices": s.indices}
                for s in self.selections
            ],
            "metadata": metadata or {},
        }
        serialized = json.dumps(config, indent=2) + "\n"
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        save_file(
            {name: layer.delta.detach().cpu().contiguous() for name, layer in self.layers.items()},
            str(destination / "patch.safetensors"),
        )
        (destination / "patch.json").write_text(serialized, encoding="utf-8")

    @classmethod
    def load(cls, model: nn.Module, directory: str | Path):
        source = Path(directory)
        config = json.loads((source / "patch.json").read_text(encoding="utf-8"))
        if (
            not isinstance(config, dict)
            or type(config.get("format_version")) is not int
            or config["format_version"] not in (1, 2)
            or config.get("method") != "paired_mlp"
        ):
            raise ValueError("unsupported paired patch format")
        train_incoming = config.get("train_incoming", True)
        if type(train_incoming) is not bool or train_incoming != (config["format_version"] == 1):
            raise ValueError("invalid paired projection mode")
        dtype_name = config.get("delta_dtype")
        dtype = _DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        selections = config.get("selections")
        if dtype is None or not isinstance(selections, list) or not selections:
            raise ValueError("invalid paired dtype or selections")
        try:
            selections = [MLPSelection(**s) for s in selections]
        except (TypeError, ValueError) as error:
            raise ValueError("invalid paired selections") from error
        checked = _layout(model, selections)
        if not train_incoming:
            checked = {name: value for name, value in checked.items() if value[1] == "input"}
        if fingerprint(model) != config.get("base_fingerprint"):
            raise ValueError("base checkpoint mismatch; use original weights and dtype")
        tensors = load_file(str(source / "patch.safetensors"))
        if set(tensors) != set(checked):
            raise ValueError("manifest and tensor keys differ")
        for name, (module, axis, indices) in checked.items():
            shape = (
                (module.out_features, len(indices))
                if axis == "input"
                else (len(indices), module.in_features)
            )
            tensor = tensors[name]
            if tensor.shape != shape or tensor.dtype != dtype or not torch.isfinite(tensor).all():
                raise ValueError("invalid paired delta shape, dtype or values")
        patch = cls(model, selections, delta_dtype=dtype, train_incoming=train_incoming)
        with torch.no_grad():
            for name, layer in patch.layers.items():
                layer.delta.copy_(tensors[name].to(layer.delta.device))
        return patch
