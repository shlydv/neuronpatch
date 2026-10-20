"""Check candidate/control scores against independently materialized gradients."""

import pytest
import torch

from experiments.fine_transfer.gradient import agreement_scores
from experiments.fine_transfer.gradient_controls import discover_controls, gradient_statistics


def test_mean_control_matches_explicit_gradients_for_unequal_prompt_lengths():
    generator = torch.Generator().manual_seed(64)
    weight = torch.randn(5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    factors, gradients = [], []
    for length in (2, 5, 3):
        inputs = torch.randn(length, 7, generator=generator, dtype=torch.float64)
        derivatives = torch.randn(length, 5, generator=generator, dtype=torch.float64)
        gradients.append(torch.autograd.grad(((inputs @ weight.T) * derivatives).sum(), weight)[0])
        factors.append((inputs, derivatives))
    stacked = torch.stack(gradients)
    mean = stacked.mean(0)
    stats = gradient_statistics(factors)
    torch.testing.assert_close(stats["mean_gradient_norm"], mean.norm(dim=0))
    torch.testing.assert_close(stats["prompt_gradient_norms"], stacked.norm(dim=1))
    torch.testing.assert_close(stats["coherence"], mean.norm(dim=0) / stacked.norm(dim=1).mean(0))
    torch.testing.assert_close(stats["worst_prompt_gain"], agreement_scores(factors))
    assert torch.all(stats["worst_prompt_gain"] <= stats["mean_gradient_norm"] + 1e-12)


def test_worst_gain_can_differ_from_mean_without_different_gradient_directions():
    # Feature 0 has a large gain on just one wording; feature 1 helps both equally.
    factors = [
        (torch.tensor([[10.0, 2.0]]), torch.ones(1, 1)),
        (torch.tensor([[1.0, 2.0]]), torch.ones(1, 1)),
    ]
    stats = gradient_statistics(factors)
    assert int(stats["mean_gradient_norm"].argmax()) == 0
    assert int(stats["worst_prompt_gain"].argmax()) == 1
    torch.testing.assert_close(stats["coherence"], torch.ones(2, dtype=torch.float64))
    # A future gain could be balancing magnitudes, not discovering semantic invariance.


def test_mean_control_handles_cancelled_and_zero_gradients_without_inventing_agreement():
    inputs, derivatives = torch.tensor([[1.0, 0.0]]), torch.ones(1, 1)
    stats = gradient_statistics([(inputs, derivatives), (inputs, -derivatives)])
    for name in ("mean_gradient_norm", "worst_prompt_gain", "coherence"):
        assert torch.equal(stats[name], torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="Finite"):
        gradient_statistics([(inputs, derivatives * float("nan"))])


def test_control_discovery_shares_frozen_model_passes_and_keeps_occurrence_budget():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(991)
    model = (
        transformers.GPTJForCausalLM(
            transformers.GPTJConfig(
                vocab_size=31,
                n_positions=32,
                n_embd=16,
                n_layer=2,
                n_head=4,
                rotary_dim=4,
                n_inner=32,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    examples = [
        (torch.tensor([[1, 2, 3, 7, 9]]), torch.tensor([7, 9])),
        (torch.tensor([[1, 4, 5, 6, 7, 9]]), torch.tensor([7, 9])),
    ]
    names = [f"transformer.h.{i}.mlp.fc_out" for i in range(2)]
    result = discover_controls(model, examples, names, neurons_per_token=3)
    assert result["backward_passes"] == 4 and result["new_fits"] == 0
    assert len(result["records"]) == 6
    assert all(len(support) == 6 for support in result["supports"].values())
    for support in result["supports"].values():
        assert all(name in names and 0 <= feature < 32 for name, feature in support)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])
        assert not parameter.requires_grad and parameter.grad is None
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(ValueError, match="same target"):
        discover_controls(
            model, [examples[0], (examples[1][0], torch.tensor([7, 8]))], names, neurons_per_token=3
        )
