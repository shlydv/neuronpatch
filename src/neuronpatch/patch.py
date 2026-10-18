"""Attach, serialize, and export a patch on an exact frozen base checkpoint."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .layers import ColumnPatch, validate_indices


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

    All original parameters are frozen. Only exact nn.Linear modules are
    supported; tied/shared target parameters and nested patches are rejected.
    """

    def __init__(self, model: nn.Module, targets: Mapping[str, Sequence[int]]):
        if not targets or any(isinstance(m, ColumnPatch) for m in model.modules()):
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
            validated[name] = validate_indices(indices, module.in_features)
        # Finish validation before changing any module or requires_grad flag.
        self.base_fingerprint = fingerprint(model)
        self.model = model
        self.targets = validated
        self.layers = {}
        model.requires_grad_(False)
        for name, indices in validated.items():
            layer = ColumnPatch(modules[name], indices)
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
            "format_version": 1,
            "method": "mlp_output_columns",
            "base_fingerprint": self.base_fingerprint,
            "targets": self.targets,
            "metadata": metadata or {},
        }
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
        if config.get("format_version") != 1 or config.get("method") != "mlp_output_columns":
            raise ValueError("unsupported patch format")
        if fingerprint(model) != config["base_fingerprint"]:
            raise ValueError("base checkpoint mismatch; use the original weights and dtype")
        tensors = load_file(str(source / "patch.safetensors"))
        if set(tensors) != set(config["targets"]):
            raise ValueError("manifest and tensor keys differ")
        for name, indices in config["targets"].items():
            module = model.get_submodule(name)
            if type(module) is not nn.Linear:
                raise ValueError(f"{name!r} is not an nn.Linear")
            validate_indices(indices, module.in_features)
            tensor = tensors[name]
            if tensor.shape != (module.out_features, len(indices)):
                raise ValueError(f"delta shape mismatch for {name}")
            if tensor.dtype != module.weight.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid delta dtype or values for {name}")
        patch = cls(model, config["targets"])
        with torch.no_grad():
            for name, layer in patch.layers.items():
                layer.delta.copy_(tensors[name])
        return patch
