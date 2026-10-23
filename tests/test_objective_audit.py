import pytest
import torch

from experiments.outcome_guard.objective_audit import compare


def test_canonical_long_alias_teaches_eos_suppression_while_event_loss_does_not():
    probabilities = torch.tensor(
        [
            [0.9, 0.03, 0.03, 0.04],
            [0.0005, 0.001, 0.0005, 0.998],
            [0.0003, 0.0003, 0.0004, 0.999],
        ],
        dtype=torch.float64,
    )
    result = compare(probabilities.log(), [[], [0], [0, 1]], [[0, 1, 3], [0, 3]], 3)
    p = result["strongest_nonterminal_eos_pressure"]
    assert p["prefix"] == [0]
    assert p["canonical_eos_gradient"] == pytest.approx(0.998 / 3)
    # Direct differentiation of -.log(.998 + .001*.999)/3.
    expected = (0.998 - 0.998 / (0.998 + 0.001 * 0.999)) / 3
    assert p["accepted_set_eos_gradient"] == pytest.approx(expected)
    assert p["accepted_set_eos_gradient"] < 0


def test_singleton_event_exactly_recovers_mean_token_ce_and_gradient_norm():
    logits = torch.tensor([[0.9, 0.1, -1], [-0.2, -0.1, 1.2]], dtype=torch.float64)
    result = compare(logits, [[], [0]], [[0, 2]], 2)
    assert result["canonical_mean_token_ce"] == pytest.approx(
        result["accepted_event_nll_per_canonical_token"]
    )
    assert result["canonical_logit_gradient_norm"] == pytest.approx(
        result["event_logit_gradient_norm"]
    )
