import math

import pytest
import torch

from neuronpatch.knowledge import knowledge_edit_loss


def test_source_scopes_kl_direction_and_weighted_components():
    probability = torch.tensor(
        [[1 / 3] * 3, [0.25, 0.5, 0.25], [0.1, 0.1, 0.8], [0.4, 0.5, 0.1]], dtype=torch.float64
    )
    logits = probability.log().requires_grad_()
    reference = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([[3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    result = knowledge_edit_loss(
        logits, reference, torch.tensor([2]), delta, alpha=2, beta=3, gamma=0.1
    )
    expected_kl = sum(math.log((1 / 3) / p) / 3 for p in (0.25, 0.5, 0.25))
    assert result.nll.item() == pytest.approx(-math.log(0.8), abs=1e-7)
    assert result.kl.item() == pytest.approx(expected_kl, abs=1e-12)
    assert result.repetition.item() == pytest.approx(-3 * math.log(0.9), abs=1e-7)
    assert result.norm.item() == 0.5
    assert result.total.item() == pytest.approx(
        -math.log(0.8) + expected_kl - 3 * math.log(0.9) + 0.5, abs=2e-7
    )
    result.total.backward()
    assert reference.grad is None
    torch.testing.assert_close(delta.grad, torch.tensor([[0.06, 0.08]], dtype=torch.float64))
    torch.testing.assert_close(
        logits.grad[2], torch.tensor([0.1, 0.1, -0.2], dtype=torch.float64), atol=1e-7, rtol=1e-6
    )


@pytest.mark.parametrize("tie", [False, True])
def test_worst_token_gradient_and_tie_distribution(tie):
    first = [0.2, 0.4, 0.4] if tie else [0.8, 0.1, 0.1]
    rows = torch.tensor(
        [[1 / 3] * 3, [1 / 3] * 3, first, [0.4, 0.2, 0.4], [1 / 3] * 3], dtype=torch.float64
    )
    logits = rows.log().requires_grad_()
    result = knowledge_edit_loss(
        logits,
        torch.zeros(2, 3, dtype=torch.float64),
        torch.tensor([0, 1]),
        torch.zeros(1, 1, dtype=torch.float64),
        alpha=0,
        beta=0,
        gamma=0,
    )
    result.total.backward()
    torch.testing.assert_close(
        logits.grad[2],
        torch.tensor([-0.4, 0.2, 0.2] if tie else [0.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        logits.grad[3],
        torch.tensor([0.2, -0.4, 0.2] if tie else [0.4, -0.8, 0.4], dtype=torch.float64),
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.count_nonzero(logits.grad[:2]) == 0
    assert torch.count_nonzero(logits.grad[-1]) == 0


@pytest.mark.parametrize("case", ["prefix", "target", "coefficient", "nan", "saturated"])
def test_invalid_or_nonfinite_objective_rejected(case):
    logits = torch.zeros(4, 3)
    reference = torch.zeros(2, 3)
    target = torch.tensor([1])
    coefficient = 1.0
    if case == "prefix":
        reference = reference[:1]
    if case == "target":
        target = torch.tensor([3])
    if case == "coefficient":
        coefficient = float("nan")
    if case == "nan":
        reference[0, 0] = float("nan")
    if case == "saturated":
        logits[-1, 1] = 10000
    with pytest.raises((ValueError, FloatingPointError)):
        knowledge_edit_loss(logits, reference, target, torch.zeros(1, 1), alpha=coefficient)
