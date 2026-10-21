"""Explicit Qwen MLP layout, matched budgets and official LoRA lifecycle."""

import torch
from torch import nn

from experiments.fine_transfer.lora_control import merge_gptj_lora, remove_gptj_lora
from neuronpatch.paired import MLPSelection


def layout(model, rank):
    groups = []
    for i, block in enumerate(model.model.layers):
        root = f"model.layers.{i}.mlp"
        names = tuple(f"{root}.{suffix}" for suffix in ("gate_proj", "up_proj", "down_proj"))
        modules = [model.get_submodule(name) for name in names]
        assert all(type(m) is nn.Linear for m in modules)
        gate, up, down = modules
        assert gate.weight.shape == up.weight.shape == down.weight.T.shape
        groups.append(
            {
                "incoming": names[:2],
                "outgoing": names[2],
                "hidden": down.out_features,
                "intermediate": down.in_features,
            }
        )
    assert groups and len({(g["hidden"], g["intermediate"]) for g in groups}) == 1
    h, m = groups[0]["hidden"], groups[0]["intermediate"]
    expected = 3 * rank * len(groups) * (h + m)
    total = (expected + 3 * h // 2) // (3 * h)
    q, remainder = divmod(total, len(groups))
    counts = [
        q + int((i + 1) * remainder // len(groups) > i * remainder // len(groups))
        for i in range(len(groups))
    ]
    assert min(counts) > 0 and max(counts) * 3 <= m
    return (
        groups,
        counts,
        {"lora": expected, "paired": 3 * h * sum(counts), "outgoing": 3 * h * sum(counts)},
    )


def supports(groups, counts, scores, *, family, seed):
    generator = torch.Generator().manual_seed(seed)
    selections = []
    for group, count, score in zip(groups, counts, scores, strict=True):
        if family == "paired_random":
            indices = torch.randperm(group["intermediate"], generator=generator)[:count]
        else:
            key = "outgoing_gradient_norm" if family == "outgoing_mean" else "paired_gradient_norm"
            k = 3 * count if family == "outgoing_mean" else count
            indices = torch.argsort(score[key], descending=True, stable=True)[:k]
        selections.append(
            MLPSelection(group["incoming"], group["outgoing"], tuple(indices.tolist()))
        )
    return selections


def attach_lora(model, names, *, rank, seed):
    from peft import LoraConfig, get_peft_model

    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        fitted = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=rank,
                lora_alpha=rank,
                lora_dropout=0.0,
                target_modules=names,
                bias="none",
                init_lora_weights=True,
            ),
        )
    for layer in fitted.modules():
        if isinstance(getattr(layer, "lora_A", None), nn.ModuleDict) and "default" in layer.lora_A:
            layer.lora_A["default"].float()
            layer.lora_B["default"].float()
    return fitted.eval()


def load_lora(model, directory, names, *, rank, seed):
    fitted = attach_lora(model, names, rank=rank, seed=seed)
    fitted.load_adapter(str(directory), adapter_name="default", is_trainable=False)
    return fitted.eval().requires_grad_(False)


# These existing helpers use only ordinary PEFT Linear APIs, not GPT-J layout.
merge_lora = merge_gptj_lora
unload_lora = remove_gptj_lora
