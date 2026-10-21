"""Official PEFT attachment for a future strong GPT-J baseline; no fitting recipe.

This is separate from the frozen, running subject-binding experiment. It does
not define training objectives, tune parameters or claim matched sparse budgets.
"""

import torch
from torch import nn


def gptj_targets(model, scope):
    """Use every block, including the final layers excluded by the FiNE selector."""
    if getattr(model, "peft_config", None) is not None:
        raise ValueError("Use an unadapted GPT-J base")
    if scope not in ("down", "attention_mlp"):
        raise ValueError("Choose down or attention_mlp")
    try:
        blocks = model.transformer.h
    except AttributeError as error:
        raise ValueError("Ordinary GPT-J blocks required") from error
    suffixes = ["mlp.fc_out"]
    if scope == "attention_mlp":
        suffixes = [f"attn.{p}_proj" for p in ("q", "k", "v", "out")]
        suffixes += ["mlp.fc_in", "mlp.fc_out"]
    names = [f"transformer.h.{i}.{suffix}" for i in range(len(blocks)) for suffix in suffixes]
    if not names:
        raise ValueError("Nonempty GPT-J blocks required")
    for name in names:
        layer = model.get_submodule(name)
        if type(layer) is not nn.Linear or not layer.weight.is_floating_point():
            raise ValueError("Materialized ordinary floating-point linear targets required")
        if layer.weight.device.type not in ("cpu", "cuda"):
            raise ValueError("This experiment helper supports CPU/CUDA targets")
    return names


def attach_gptj_lora(model, *, rank=2, scope="attention_mlp", seed=43):
    """Attach standard zero-B LoRA, scale1, dropout0 and FP32 trainable adapters.

    Base weights remain in their original dtype. This deliberately gives the
    comparator broad coverage and reports its larger actual scalar budget.
    Trainable factor dtype is explicit across old and new PEFT versions.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if type(rank) is not int or rank <= 0:
        raise ValueError("Positive integer rank required")
    names = gptj_targets(model, scope)
    expected = rank * sum(
        model.get_submodule(name).in_features + model.get_submodule(name).out_features
        for name in names
    )
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        wrapped = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=rank,
                lora_dropout=0.0,
                target_modules=names,
                bias="none",
                init_lora_weights=True,
                use_rslora=False,
                use_dora=False,
            ),
        )
    adapted = 0
    for module in wrapped.modules():
        a, b = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
        if isinstance(a, nn.ModuleDict) and "default" in a:
            a["default"].float()
            b["default"].float()
            adapted += 1
    assert adapted == len(names)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert trainable == expected
    assert all(p.dtype == torch.float32 for p in wrapped.parameters() if p.requires_grad)
    return wrapped.eval(), {
        "scope": scope,
        "rank": rank,
        "alpha": rank,
        "dropout": 0.0,
        "seed": seed,
        "targets": names,
        "trainable_scalars": trainable,
        "trainable_dtype": "float32",
        "initialization": "PEFT kaiming A, zero B",
        "status": "Attachment/integrity only; no full-size fit or frozen comparison protocol",
    }


def load_gptj_lora(model, directory):
    """Reload the control without rounding FP32 factors into an FP16 base.

    Older PEFT constructs factors in the base dtype before copying saved values.
    Construct FP32 factors before official load_adapter copies those tensors.
    Base revision/fingerprint validation belongs to the experiment runner.
    """
    from peft import LoraConfig

    config = LoraConfig.from_pretrained(str(directory))
    if isinstance(config.target_modules, str):
        raise ValueError("Use explicit targets from this standard GPT-J control")
    # New PEFT can shorten saved target names to equivalent suffixes.
    matched = {
        name
        for name, _ in model.named_modules()
        if any(name == p or name.endswith("." + p) for p in config.target_modules)
    }
    scopes = [
        scope for scope in ("down", "attention_mlp") if set(gptj_targets(model, scope)) == matched
    ]
    if (
        len(scopes) != 1
        or config.lora_alpha != config.r
        or config.lora_dropout != 0
        or config.bias != "none"
        or config.use_dora
        or config.use_rslora
        or config.fan_in_fan_out
        or config.modules_to_save
    ):
        raise ValueError("Use an adapter exported by this standard GPT-J control")
    wrapped, _ = attach_gptj_lora(model, rank=config.r, scope=scopes[0])
    wrapped.load_adapter(str(directory), adapter_name="default", is_trainable=False)
    return wrapped.eval().requires_grad_(False)


@torch.no_grad()
def merge_gptj_lora(wrapped):
    """Merge one projection at a time, preserving original base storage dtypes.

    Older PEFT promotes FP16 base weights when adding FP32 factors. Casting only
    after all projections merge could double large-model memory and break input
    dtypes. Merge and restore each base layer's dtype, then unload the wrappers.
    This mutates the supplied model, as ordinary PEFT merging does.
    """
    layers = [
        m
        for m in wrapped.modules()
        if isinstance(getattr(m, "lora_A", None), nn.ModuleDict) and "default" in m.lora_A
    ]
    if not layers:
        raise ValueError("Expected an attached standard LoRA control")
    for layer in layers:
        base = layer.get_base_layer()
        dtype = base.weight.dtype
        layer.merge(safe_merge=True)
        base.to(dtype=dtype)
        if not torch.isfinite(base.weight).all():
            raise FloatingPointError("Merged weights overflowed their original dtype")
    return wrapped.unload().eval()
