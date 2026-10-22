"""Nominal pre-quantization weight-update norms,not raw LoRA-factor norms."""

import json
import math

from safetensors.torch import load_file


def nominal_update_norm(directory, family):
    if family != "lora":
        tensors = load_file(str(directory / "patch.safetensors"))
        return math.sqrt(sum(float(t.double().square().sum()) for t in tensors.values()))
    config = json.loads((directory / "adapter_config.json").read_text())
    assert not config.get("use_dora", False) and not config.get("use_rslora", False)
    assert config.get("bias", "none") == "none"
    scale = config["lora_alpha"] / config["r"]
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    names = [n for n in tensors if n.endswith("lora_A.weight")]
    assert len(tensors) == 2 * len(names) and names
    squared = 0.0
    for name in names:
        a = tensors[name].double()
        b = tensors[name.replace("lora_A.weight", "lora_B.weight")].double()
        assert a.shape[0] == b.shape[1] == config["r"]
        # ||BA||_F² = trace((BᵀB)(AAᵀ));avoid forming model-sized dense updates.
        squared += float(((b.T @ b) * (a @ a.T)).sum()) * scale**2
    assert math.isfinite(squared) and squared >= -1e-12
    return math.sqrt(max(0.0, squared))
