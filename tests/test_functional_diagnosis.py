import hashlib
import json
import sys

import pytest
import torch
from safetensors.torch import save_file

from experiments.functional_selection.diagnose import diagnose, main, select, summarize


def test_joint_risk_distinguishes_cancellation_from_individual_sensitivity():
    gain = torch.tensor([2.0, 1.0]).double()
    response = torch.tensor([[4.0, -4.0], [-4.0, 4.0]]).double()
    both = summarize(gain, response, torch.tensor([True, True]))
    assert both["joint_curvature"] == 0 and both["diagonal_curvature"] == 32
    assert both["predicted_gain_squared_per_joint_curvature"] is None
    assert both["zero_measured_joint_curvature"]
    one = summarize(gain, response, torch.tensor([True, False]))
    assert one["joint_curvature"] == 16
    assert one["predicted_gain_squared_per_joint_curvature"] == 0.25


def fixture(responses):
    gain = torch.tensor([2.0, 1.9]).double()
    norms = torch.ones(2).double()
    eligible = torch.ones(2, dtype=torch.bool)
    full = select(gain, norms, responses, eligible, [2], [1], 1.0)
    return {
        "gain": gain,
        "direction_squared_norm": norms,
        "responses": responses,
        "eligible": eligible,
        "selected/gain_risk": full,
        "selected/adam_gain": torch.tensor([True, False]),
    }


def test_anchor_holdout_exposes_an_update_with_zero_training_and_large_unseen_risk():
    # One feature influences only anchor A; the other only anchor B. Removing
    # A makes the first feature look costless even though its held-out risk is400.
    responses = torch.tensor([[20, 0], [20, 0], [0, 20], [0, 20], [0, 0], [0, 0]]).double()
    result = diagnose(fixture(responses), [2], [1], probes_per_anchor=2, damping_ratio=1.0)
    first = result["leave_one_anchor_out"][0]
    assert first["training"]["selected"] == [0]
    assert first["training"]["joint_curvature"] == 0
    assert first["heldout"]["joint_curvature"] == 400
    assert set(first["training_probe_rows"]).isdisjoint(first["heldout_probe_rows"])
    second = result["leave_one_anchor_out"][1]
    assert second["training"]["selected"] == [1]
    assert second["heldout"]["joint_curvature"] == 400


def test_independent_noise_halves_can_select_disjoint_apparently_safe_supports():
    responses = torch.tensor([[10, 0], [0, 10]] * 3).double()
    result = diagnose(fixture(responses), [2], [1], probes_per_anchor=2, damping_ratio=1.0)
    assert result["noise_split_common_neurons"] == 0
    assert result["noise_split_union_neurons"] == 2
    for fold in result["leave_one_probe_per_anchor_out"]:
        assert fold["training"]["joint_curvature"] == 0
        assert fold["heldout"]["joint_curvature"] == 100


def test_diagnostic_selection_keeps_layer_budgets_and_excludes_random_only_candidates():
    gain = torch.tensor([5, 4, 100, 3, 2, 1]).double()
    eligible = torch.tensor([True, True, False, True, True, True])
    mask = select(gain, torch.ones(6), torch.ones(4, 6), eligible, [3, 3], [1, 2], 1.0)
    assert mask.tolist() == [True, False, False, True, True, False]


def test_cli_binds_saved_inputs_to_verification_and_keeps_zero_risk_json_finite(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    protocol = {
        "expected_one_step_edits": 16,
        "conditions": ["gradient_norm", "adam_gain", "gain_risk", "random"],
        "scales": [0.01, 0.03],
        "fisher_probes_per_anchor": 2,
        "damping_ratio": 1.0,
    }
    (results / "protocol.json").write_text(json.dumps(protocol))
    (results / "completion.json").write_text(json.dumps({"status": "complete"}))
    (results / "identity.json").write_text(json.dumps({"layer_counts": [1]}))
    for case in (0, 1):
        (results / f"case-{case}-discovery.json").write_text(
            json.dumps({"case": case, "candidates": [{"indices": [0, 1]}]})
        )
        save_file(
            fixture(torch.tensor([[10, 0], [0, 10]] * 3).double()),
            results / f"case-{case}-selection.safetensors",
        )
    proof = tmp_path / "integrity.json"
    proof.write_text(
        json.dumps(
            {
                "complete": True,
                "file_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in results.iterdir()
                },
            }
        )
    )
    output = tmp_path / "diagnosis.json"
    command = [
        "diagnose",
        "--results",
        str(results),
        "--integrity",
        str(proof),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", command)
    main()
    decoded = json.loads(output.read_text())
    assert decoded["post_hoc_diagnostic"] and decoded["new_model_executions"] == 0
    assert len(decoded["cases"]) == 2
    assert all(c["noise_split_common_neurons"] == 0 for c in decoded["cases"])
    assert "NaN" not in output.read_text() and "Infinity" not in output.read_text()
    # Adding an otherwise irrelevant metadata field must invalidate the old
    # verifier binding before the diagnostic reads any result tensors.
    (results / "identity.json").write_text(json.dumps({"layer_counts": [1], "changed": True}))
    rejected = tmp_path / "rejected.json"
    monkeypatch.setattr(sys, "argv", [*command[:-1], str(rejected)])
    with pytest.raises(AssertionError, match="Changed verified input"):
        main()
    assert not rejected.exists()
