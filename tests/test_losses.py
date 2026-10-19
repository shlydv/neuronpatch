import pytest
import torch

from neuronpatch.losses import preservation_kl


def test_forward_kl_value_and_analytical_gradient():
    edited = torch.tensor([[[0.2, -0.3, 0.5]]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[[0.7, 0.1, -0.2]]], dtype=torch.float64, requires_grad=True)
    p = edited.detach().softmax(-1)
    q = reference.detach().softmax(-1)
    loss = preservation_kl(edited, reference, torch.tensor([[True]]))
    torch.testing.assert_close(loss, (q * (q.log() - p.log())).sum())
    loss.backward()
    torch.testing.assert_close(edited.grad, p - q)
    assert reference.grad is None


def test_reverse_kl_gradient_matches_independent_autograd_reference():
    edited = torch.tensor([[[0.2, -0.3, 0.5]]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[[0.7, 0.1, -0.2]]], dtype=torch.float64, requires_grad=True)
    p = edited.softmax(-1)
    expected = (p * (p.log() - reference.detach().log_softmax(-1))).sum()
    actual = preservation_kl(
        edited, reference, torch.tensor([[True]]), direction="edited_to_reference"
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, edited)[0], torch.autograd.grad(expected, edited)[0]
    )


def test_masked_nans_and_equal_example_weighting():
    edited = torch.tensor(
        [[[0.0, 1.0], [float("nan"), float("nan")]], [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )
    reference = torch.zeros_like(edited, requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])
    actual = preservation_kl(edited, reference, mask)
    expected = (
        preservation_kl(edited[:1], reference[:1], mask[:1])
        + preservation_kl(edited[1:], reference[1:], mask[1:])
    ) / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(edited.grad).all()
    assert torch.count_nonzero(edited.grad[0, 1]) == 0
    assert reference.grad is None


def test_identical_distributions_have_zero_loss_and_gradient():
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    for direction in ("reference_to_edited", "edited_to_reference"):
        loss = preservation_kl(
            x, x.detach(), torch.ones(2, 3, dtype=torch.bool), direction=direction
        )
        torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(
            torch.autograd.grad(loss, x)[0], torch.zeros_like(x), atol=1e-12, rtol=0
        )


def test_rejects_empty_examples_and_nonfinite_selected_logits():
    x = torch.zeros(2, 3, 5)
    with pytest.raises(ValueError):
        preservation_kl(x, x, torch.tensor([[True, False, False], [False, False, False]]))
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        preservation_kl(x, x, torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("direction", ["reference_to_edited", "edited_to_reference"])
def test_max_token_loss_matches_independent_reference_and_masked_gradients(direction):
    edited = torch.tensor(
        [
            [[4.0, 0.0], [1.0, 0.0], [float("nan"), float("nan")]],
            [[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    reference = torch.zeros_like(edited, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    actual = preservation_kl(edited, reference, mask, direction=direction, token_reduction="max")
    # Both rows have a unique maximum, at (0,0) and (1,1). Compute only those
    # positions independently, then check masked/nonmax positions get no gradient.
    logits = torch.stack([edited[0, 0], edited[1, 1]])
    p = logits.softmax(-1)
    q = torch.full_like(p, 0.5)
    expected = (
        (
            (q * (q.log() - p.log()))
            if direction == "reference_to_edited"
            else (p * (p.log() - q.log()))
        )
        .sum(-1)
        .mean()
    )
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual, edited)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(expected, edited)[0])
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[0, 1:]) == 0
    assert torch.count_nonzero(grad[1, [0, 2]]) == 0
    assert reference.grad is None


def test_max_token_ties_and_identity_are_finite_and_explicit():
    x = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]], dtype=torch.double, requires_grad=True)
    reference = torch.zeros_like(x)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = preservation_kl(x, reference, mask, token_reduction="max")
    grad = torch.autograd.grad(loss, x)[0]
    torch.testing.assert_close(grad[0, 0], grad[0, 1])
    identity = preservation_kl(x, x.detach(), mask, token_reduction="max")
    torch.testing.assert_close(identity, torch.zeros_like(identity), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(identity, x)[0], torch.zeros_like(x), atol=1e-12, rtol=0
    )
    with pytest.raises(ValueError, match="token reduction"):
        preservation_kl(x, reference, mask, token_reduction="sum")
