"""Explicit GPT-J update groups; no attribution ranking or adapter refitting."""

from collections import Counter

import torch
from torch import nn


def conditions(names, layers):
    """Return all keep-only/remove-group conditions before observing outputs.

    The three projection families partition every adapted block. The final-three
    layer control overlaps those families and is reported separately.
    """
    if type(layers) is not int or layers < 4:
        raise ValueError("At least four GPT-J blocks required")
    families = {
        "attention": [f"attn.{p}_proj" for p in ("q", "k", "v", "out")],
        "incoming": ["mlp.fc_in"],
        "outgoing": ["mlp.fc_out"],
    }
    groups = {
        family: {f"transformer.h.{i}.{suffix}" for i in range(layers) for suffix in suffixes}
        for family, suffixes in families.items()
    }
    expected = set().union(*groups.values())
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError("Exactly all six adapted GPT-J projections per block required")
    groups["final-three"] = {name for name in names if int(name.split(".")[2]) >= layers - 3}
    result = [{"id": "full", "keep": list(names)}]
    for group, members in groups.items():
        for action, keep in (("only", members), ("without", expected - members)):
            result.append(
                {"id": f"{action}-{group}", "keep": [name for name in names if name in keep]}
            )
    return result


@torch.no_grad()
def retain_updates(wrapped, names, keep):
    """Destructively zero excluded LoRA B factors before the normal FP16 merge.

    Call only on a freshly reloaded frozen adapter. Included A and B factors are
    unchanged. Zeroing B removes the complete learned update, including changes
    to A. This does not zero base activations, heads, or neurons. The same group
    intervention is used for every prompt and token.
    """
    if len(set(names)) != len(names) or len(set(keep)) != len(keep):
        raise ValueError("Unique projection names required")
    if not names or not set(keep) <= set(names):
        raise ValueError("Retained projections must be a subset of the full adapter")
    adapted = {}
    for name, layer in wrapped.named_modules():
        a = getattr(layer, "lora_A", None)
        if not isinstance(a, nn.ModuleDict):
            continue
        b = layer.lora_B
        if set(a) != {"default"} or set(b) != {"default"}:
            raise ValueError("Exactly one default adapter required")
        target = name.removeprefix("base_model.model.")
        adapted[target] = layer
    if set(adapted) != set(names) or any(p.requires_grad for p in wrapped.parameters()):
        raise ValueError("A frozen full GPT-J adapter is required")
    # Validate every layer before mutating any factor.
    for layer in adapted.values():
        factors = (layer.lora_A["default"].weight, layer.lora_B["default"].weight)
        if layer.scaling["default"] != 1 or any(
            p.dtype != torch.float32 or not torch.isfinite(p).all() for p in factors
        ):
            raise ValueError("Finite FP32 factors with scale one required")
    sizes = Counter()
    for name, layer in adapted.items():
        size = sum(
            p.numel() for p in (layer.lora_A["default"].weight, layer.lora_B["default"].weight)
        )
        if name in keep:
            sizes["retained_factor_scalars"] += size
        else:
            layer.lora_B["default"].weight.zero_()
            sizes["excluded_factor_scalars"] += size
    return {
        "keep": list(keep),
        "excluded": [name for name in names if name not in keep],
        "retained_factor_scalars": sizes["retained_factor_scalars"],
        "excluded_factor_scalars": sizes["excluded_factor_scalars"],
        "new_trainable_scalars": 0,
        "new_training_steps": 0,
    }
