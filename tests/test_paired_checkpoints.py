import json
import math

import torch

from experiments.paired_checkpoints.observe import calibration


def test_lora_effective_update_norm_matches_dense_product_with_scaling(tmp_path):
    from safetensors.torch import save_file

    from experiments.paired_checkpoints.geometry import nominal_update_norm

    a = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
    b = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [3.0, 2.0]])
    save_file(
        {"layer.lora_A.weight": a, "layer.lora_B.weight": b}, tmp_path / "adapter_model.safetensors"
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "bias": "none"})
    )
    expected = float((2 * (b.double() @ a.double())).norm())
    assert math.isclose(nominal_update_norm(tmp_path, "lora"), expected, rel_tol=1e-12)


def test_confidence_gate_requires_each_token_including_eos_not_only_mean_loss():
    probabilities = torch.tensor(
        [[0.0005, 0.999, 0.0005], [0.1, 0.05, 0.85], [0.005, 0.99, 0.005]], dtype=torch.float64
    )
    teacher = {
        "logits": probabilities.log(),
        "labels": torch.tensor([1, 2, 1]),
        "lengths": torch.tensor([2, 1]),
    }
    result = calibration(teacher, 1, 0.9)
    assert result["task_mean_nll"] < -math.log(0.9)
    assert result["task_worst_token_nll"] > -math.log(0.9)
    assert not result["adequate"]
    teacher["logits"][1] = torch.tensor([0.005, 0.005, 0.99], dtype=torch.float64).log()
    assert calibration(teacher, 1, 0.9)["adequate"]
