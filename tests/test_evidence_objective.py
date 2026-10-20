"""Mechanism checks for the experimental evidence objective, not LLM quality."""

import runpy
from pathlib import Path

import pytest
import torch

functions = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/evidence_pairs/objective.py")
)
evidence_scores, separation_loss = (
    functions[name] for name in ("evidence_scores", "separation_loss")
)


def fixture():
    generator = torch.Generator().manual_seed(91)
    logits = torch.randn(5, 2, 7, generator=generator, dtype=torch.float64)
    answers = torch.tensor([1, 3, 4, 5, 6])
    return logits, answers


def test_vocabulary_bias_cannot_improve_loss_from_reference():
    logits, answers = fixture()
    # Even question-specific vocabulary biases cancel when shared within a pair.
    bias = (
        torch.randn(5, 1, 7, generator=torch.Generator().manual_seed(27), dtype=torch.float64) * 5
    )
    reference = evidence_scores(logits, answers, unknown_token=0)
    edited = evidence_scores(logits + bias, answers, unknown_token=0)
    torch.testing.assert_close(edited[:, 1] - edited[:, 0], reference[:, 1] - reference[:, 0])
    torch.testing.assert_close(
        separation_loss(edited, reference, center_weight=0),
        separation_loss(reference, reference, center_weight=0),
    )
    assert separation_loss(edited, reference) > separation_loss(reference, reference)


def test_pair_gradient_uses_context_and_detaches_reference():
    logits, answers = fixture()
    reference = evidence_scores(logits, answers, unknown_token=0).requires_grad_(True)
    scores = reference.detach().clone().requires_grad_(True)
    separation_loss(scores, reference).backward()
    assert reference.grad is None
    assert (scores.grad[:, 0] > 0).all() and (scores.grad[:, 1] < 0).all()
    torch.testing.assert_close(scores.grad.sum(-1), torch.zeros(5, dtype=torch.float64))
    improved = scores.detach() + torch.tensor([-0.1, 0.1])
    assert separation_loss(improved, reference) < separation_loss(scores, reference)


def test_first_token_gradient_matches_numerical_derivative():
    logits, answers = fixture()
    logits.requires_grad_(True)
    reference = evidence_scores(logits.detach(), answers, unknown_token=0)
    assert torch.autograd.gradcheck(
        lambda value: separation_loss(evidence_scores(value, answers, unknown_token=0), reference),
        logits,
    )


def test_half_precision_inputs_use_float_loss_and_finite_gradients():
    logits, answers = fixture()
    logits = logits.half().requires_grad_(True)
    scores = evidence_scores(logits, answers, unknown_token=0)
    loss = separation_loss(scores, scores.detach())
    assert scores.dtype == loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "change",
    [
        lambda x, a: (x[:, :1], a),
        lambda x, a: (x[:0], a[:0]),
        lambda x, a: (x.long(), a),
        lambda x, a: (x * torch.nan, a),
        lambda x, a: (x, a.float()),
        lambda x, a: (x, a * 0),
        lambda x, a: (x, a * 0 - 1),
    ],
)
def test_invalid_pair_logits_rejected(change):
    logits, answers = change(*fixture())
    with pytest.raises(ValueError):
        evidence_scores(logits, answers, unknown_token=0)


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0}, {"center_weight": -1}, {"margin": float("nan")}]
)
def test_loss_configuration_validation(kwargs):
    value = torch.zeros(1, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        separation_loss(value, value, **kwargs)
