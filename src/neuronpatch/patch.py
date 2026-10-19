"""Attach, serialize, and export a patch on an exact frozen base checkpoint."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .grouped import GroupedRowPatch, validate_groups
from .layers import ColumnPatch, RowPatch, validate_indices


def fingerprint(model: nn.Module) -> str:
    """Hash names, shapes, dtypes and bytes of parameters and persistent buffers.

    This streams one tensor at a time to CPU. It is deliberately strict and can
    be expensive on large models. Architecture/tokenizer identity is additional
    caller metadata, not something a weight hash proves.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


class NeuronPatch:
    """Own patches attached in-place to a model; optimize trainable_parameters().

    All original parameters are frozen. axis="input" updates selected weight
    columns (default); axis="output" updates selected weight rows. One axis applies
    to all targets. Only exact nn.Linear modules are supported; tied/shared target
    parameters and nested patches are rejected. Optional groups maps every output
    target to one group label per selected row; members share one additive delta.
    Labels must be contiguous from zero within each target. Group discovery is a
    caller decision, not an inference made by this class.
    """

    def __init__(
        self,
        model: nn.Module,
        targets: Mapping[str, Sequence[int]],
        *,
        axis: str = "input",
        groups: Mapping[str, Sequence[int]] | None = None,
    ):
        if axis not in ("input", "output"):
            raise ValueError("axis must be input (columns) or output (rows)")
        self.axis = axis
        if groups is not None and (axis != "output" or set(groups) != set(targets)):
            raise ValueError("groups require output rows and exactly the target module keys")
        if not targets or any(
            isinstance(m, (ColumnPatch, RowPatch, GroupedRowPatch)) for m in model.modules()
        ):
            raise ValueError("provide targets on an unpatched model")
        modules = dict(model.named_modules(remove_duplicate=False))
        parameters = list(model.named_parameters(remove_duplicate=False))
        validated = {}
        for name, indices in targets.items():
            module = modules.get(name)
            if not name or type(module) is not nn.Linear:
                raise ValueError(f"{name!r} must name an ordinary nn.Linear submodule")
            if sum(m is module for m in modules.values()) != 1:
                raise ValueError(f"shared module {name!r} is unsupported")
            for parameter in module.parameters():
                if sum(p is parameter for _, p in parameters) != 1:
                    raise ValueError(f"tied parameter in {name!r} is unsupported")
            if module.weight.device.type == "meta" or not module.weight.is_floating_point():
                raise ValueError("targets must be materialized floating-point layers")
            validated[name] = validate_indices(
                indices, module.in_features if axis == "input" else module.out_features
            )
        validated_groups = (
            {
                name: validate_groups(groups[name], len(indices))
                for name, indices in validated.items()
            }
            if groups is not None
            else None
        )
        # Finish validation before changing any module or requires_grad flag.
        self.base_fingerprint = fingerprint(model)
        self.model = model
        self.targets = validated
        self.groups = validated_groups
        self.layers = {}
        model.requires_grad_(False)
        for name, indices in validated.items():
            layer = (
                GroupedRowPatch(modules[name], indices, validated_groups[name])
                if validated_groups is not None
                else (ColumnPatch if axis == "input" else RowPatch)(modules[name], indices)
            )
            layer.train(modules[name].training)
            model.set_submodule(name, layer)
            self.layers[name] = layer

    def trainable_parameters(self):
        return (layer.delta for layer in self.layers.values())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def merged_model(self) -> nn.Module:
        """Return a deep-copied ordinary model with deltas folded into weights.

        Requires memory for a second model. Floating-point accumulation order
        may cause small differences from the unmerged forward pass.
        """
        result = deepcopy(self.model)
        for name in self.targets:
            result.set_submodule(name, result.get_submodule(name).merged_linear())
        return result

    def save(self, directory: str | Path, *, metadata: dict | None = None) -> None:
        """Write only compact deltas and a manifest to a new directory."""
        config = {
            "format_version": 1 if self.axis == "input" else 2,
            "method": "mlp_output_columns" if self.axis == "input" else "linear_output_rows",
            "base_fingerprint": self.base_fingerprint,
            "targets": self.targets,
            "metadata": metadata or {},
        }
        if self.groups is not None:
            config.update(format_version=3, method="linear_grouped_output_rows", groups=self.groups)
        manifest = json.dumps(config, indent=2) + "\n"
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        tensors = {
            name: layer.delta.detach().cpu().contiguous() for name, layer in self.layers.items()
        }
        save_file(tensors, str(destination / "patch.safetensors"))
        (destination / "patch.json").write_text(manifest, encoding="utf-8")

    @classmethod
    def load(cls, model: nn.Module, directory: str | Path) -> "NeuronPatch":
        """Validate an artifact and attach it to an identical, unpatched base."""
        source = Path(directory)
        config = json.loads((source / "patch.json").read_text(encoding="utf-8"))
        formats = {
            (1, "mlp_output_columns"): "input",
            (2, "linear_output_rows"): "output",
            (3, "linear_grouped_output_rows"): "output",
        }
        axis = formats.get((config.get("format_version"), config.get("method")))
        if axis is None:
            raise ValueError("unsupported patch format")
        groups = config.get("groups")
        if config["format_version"] == 3:
            if not isinstance(groups, dict) or set(groups) != set(config["targets"]):
                raise ValueError("group manifest must match target keys")
        elif groups is not None:
            raise ValueError("legacy patch format does not support groups")
        if fingerprint(model) != config["base_fingerprint"]:
            raise ValueError("base checkpoint mismatch; use the original weights and dtype")
        tensors = load_file(str(source / "patch.safetensors"))
        if set(tensors) != set(config["targets"]):
            raise ValueError("manifest and tensor keys differ")
        for name, indices in config["targets"].items():
            module = model.get_submodule(name)
            if type(module) is not nn.Linear:
                raise ValueError(f"{name!r} is not an nn.Linear")
            validate_indices(
                indices, module.in_features if axis == "input" else module.out_features
            )
            assignments = (
                validate_groups(groups[name], len(indices)) if groups is not None else None
            )
            tensor = tensors[name]
            shape = (
                (module.out_features, len(indices))
                if axis == "input"
                else (
                    len(set(assignments)) if assignments is not None else len(indices),
                    module.in_features,
                )
            )
            if tensor.shape != shape:
                raise ValueError(f"delta shape mismatch for {name}")
            if tensor.dtype != module.weight.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid delta dtype or values for {name}")
        patch = cls(model, config["targets"], axis=axis, groups=groups)
        with torch.no_grad():
            for name, layer in patch.layers.items():
                layer.delta.copy_(tensors[name])
        return patch
