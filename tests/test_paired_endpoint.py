import math

import pytest
import torch

from experiments.paired_endpoint.run import comparison, verify_parent


def test_endpoint_divergence_matches_two_categorical_distributions():
    live = {
        "logits": torch.tensor([[math.log(0.2), math.log(0.8)]], dtype=torch.float64),
        "labels": torch.tensor([1]),
        "lengths": torch.tensor([1]),
    }
    merged = {**live, "logits": torch.tensor([[math.log(0.6), math.log(0.4)]], dtype=torch.float64)}
    result = comparison(live, merged)
    assert not result["logits_exact"] and result["changed_teacher_argmax_positions"] == 1
    assert result["mean_kl_live_to_merged"] == pytest.approx(
        0.2 * math.log(0.2 / 0.6) + 0.8 * math.log(0.8 / 0.4)
    )
    assert result["mean_kl_merged_to_live"] == pytest.approx(
        0.6 * math.log(0.6 / 0.2) + 0.4 * math.log(0.4 / 0.8)
    )
    assert comparison(live, live)["mean_kl_live_to_merged"] == 0


def test_replay_rejects_changed_frozen_inputs_before_inference(tmp_path):
    (tmp_path / "inputs.json").write_text("changed")
    with pytest.raises(AssertionError, match="Changed frozen parent file"):
        verify_parent(tmp_path, {"parent_files_sha256": {"inputs.json": "a" * 64}})
