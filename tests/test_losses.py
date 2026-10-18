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
