from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.functional_selection.directions import (
    DirectionBank,
    build_directions,
    probe_dots,
    rank_scores,
)
from experiments.functional_selection.fisher import (
    capture_fisher_factors,
    logit_weights,
    restrict_factors,
)
from experiments.functional_selection.perturb import perturb
from experiments.paired_mlp.selection import capture_response_factors
from neuronpatch.losses import completion_nll
from neuronpatch.paired import MLPSelection
from neuronpatch.supervision import ResponseExample


class CausalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.up = nn.Linear(4, 7, bias=False)
        self.gate = nn.Linear(4, 7, bias=False)
        self.down = nn.Linear(7, 4, bias=False)
        self.head = nn.Linear(4, 9, bias=False)
        self.fail = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        x = x.cumsum(1)
        up = self.up(x)
        if self.fail:
            raise RuntimeError("fixture failure after an attached projection executes")
        hidden = self.down(up * F.silu(self.gate(x)))
        return SimpleNamespace(logits=self.head(hidden.cumsum(1)))


def example(ids, start):
    ids = torch.tensor([ids])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, start:] = True
    return ResponseExample(ids, torch.ones_like(mask), mask, start)


def test_fisher_weights_match_exact_kl_hessian_while_reference_kl_gradient_is_zero():
    logits = torch.tensor([[0.7, -0.1, 0.3], [-0.3, 0.4, 1.0]], dtype=torch.float64)
    logits.requires_grad_(True)
    reference = logits.detach().log_softmax(-1)

    def drift(values):
        return F.kl_div(values.log_softmax(-1), reference, log_target=True, reduction="sum") / 2

    torch.testing.assert_close(
        torch.autograd.grad(drift(logits), logits)[0], torch.zeros_like(logits), atol=1e-16, rtol=0
    )
    expected = torch.autograd.functional.hessian(drift, logits).reshape(6, 6)
    # A complete orthogonal set has exactly identity second moment; no noisy
    # Monte Carlo convergence threshold is used for this covariance oracle.
    noise = torch.eye(6, dtype=torch.float64).reshape(6, 2, 3) * (6**0.5)
    weights = torch.stack([logit_weights(logits, n).flatten() for n in noise])
    assert not weights.requires_grad
    observed = weights.T @ weights / len(weights)
    torch.testing.assert_close(observed, expected, atol=1e-15, rtol=1e-14)
    assert observed.norm() > 0.1


@pytest.mark.parametrize("kind", ["gradient", "adam_first"])
def test_directions_and_compact_fisher_probes_match_dense_weight_oracles(kind):
    torch.manual_seed(21)
    base = CausalMLP().eval().requires_grad_(False)
    reference = deepcopy(base).requires_grad_(True)
    selections = [MLPSelection(("up", "gate"), "down", (5, 0, 2))]
    names = ("up", "gate", "down")
    samples = [example([1, 2, 4, 6, 2], 3), example([1, 3, 7, 2], 2)]
    captures = [capture_response_factors(base, s, names)[0] for s in samples]
    restricted = restrict_factors(captures, selections)
    bank = build_directions(restricted, selections, kind=kind)
    losses = [
        completion_nll(
            reference(input_ids=s.input_ids).logits,
            s.input_ids,
            s.completion_mask,
            s.attention_mask,
        )
        for s in samples
    ]
    weights = [reference.get_submodule(name).weight for name in names]
    gradients = torch.autograd.grad(torch.stack(losses).mean(), weights)
    indices = torch.tensor(selections[0].indices)
    dense, squared = {}, torch.zeros(3, dtype=torch.float64)
    for name, parameter, gradient in zip(names, weights, gradients, strict=True):
        g = gradient.double()
        if kind == "gradient":
            update = -g
        else:
            # Actual fresh PyTorch Adam provides a separate optimizer oracle.
            p = nn.Parameter(torch.zeros_like(parameter, dtype=torch.float64))
            optimizer = torch.optim.Adam([p], lr=1.0)
            p.grad = g.clone()
            optimizer.step()
            update = p.detach()
        axis = 0 if name != "down" else 1
        dense[name] = update.index_select(axis, indices)
        squared += dense[name].square().sum(1 - axis)
    for name in names:
        scale = squared.sqrt()
        expected = dense[name] / (scale[:, None] if name != "down" else scale[None, :])
        torch.testing.assert_close(bank.matrices[name], expected, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(bank.squared_norm, torch.ones(3).double(), atol=1e-14, rtol=0)

    normal = example([1, 4, 5, 3, 2], 3)
    rng = torch.get_rng_state().clone()
    probes = capture_fisher_factors(base, normal, selections, probes=3, seed=8)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    assert all(not m._forward_hooks for m in base.modules())
    values = reference(input_ids=normal.input_ids).logits[0, :-1][normal.completion_mask[0, 1:]]
    generator = torch.Generator().manual_seed(8)
    for i, probe in enumerate(probes):
        noise = torch.empty_like(values, dtype=torch.float32)
        noise.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        coefficients = logit_weights(values, noise)
        grads = torch.autograd.grad(
            (values * coefficients).sum(), weights, retain_graph=i < len(probes) - 1
        )
        expected_dots = torch.zeros(3, dtype=torch.float64)
        for name, gradient in zip(names, grads, strict=True):
            axis = 0 if name != "down" else 1
            wanted = gradient.index_select(axis, indices)
            inputs, derivative = probe[name]
            torch.testing.assert_close(derivative.T @ inputs, wanted, atol=2e-6, rtol=2e-5)
            expected_dots += (wanted.double() * bank.matrices[name]).sum(1 - axis)
        torch.testing.assert_close(probe_dots(bank, probe), expected_dots, atol=2e-6, rtol=2e-5)


def test_gain_risk_score_can_reject_the_largest_gradient_and_reports_cross_terms():
    bank = DirectionBank(
        (), {}, torch.tensor([3.0, 2.0]).double(), torch.ones(2).double(), "fixture"
    )
    result = rank_scores(bank, torch.tensor([[10.0, 0.0], [-10.0, 0.0]]))
    assert bank.gain.argmax() == 0 and result["scores"].argmax() == 1
    canceled = rank_scores(bank, torch.tensor([[1.0, -1.0]]))
    assert canceled["joint_curvature_all_candidates"] == 0
    assert canceled["diagonal_curvature_all_candidates"] == 2
    undetected = rank_scores(bank, torch.zeros(2, 2))
    assert not undetected["risk_informative"]
    assert torch.isfinite(undetected["scores"]).all()


def test_failed_fisher_forward_removes_hooks_and_does_not_change_parameters():
    model = CausalMLP().eval().requires_grad_(False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.fail = True
    with pytest.raises(RuntimeError, match="fixture failure"):
        capture_fisher_factors(
            model,
            example([1, 3, 2], 2),
            [MLPSelection(("up", "gate"), "down", (0, 2))],
            probes=1,
            seed=2,
        )
    assert all(not m._forward_hooks for m in model.modules())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_finite_step_restores_exact_slices_on_exception_and_rejects_overflow_before_mutation():
    torch.manual_seed(85)
    model = CausalMLP().half().eval().requires_grad_(False)
    selections = [MLPSelection(("up", "gate"), "down", (5, 0, 2))]
    factors = capture_response_factors(model, example([1, 4, 5, 2], 2), ("up", "gate", "down"))[0]
    bank = build_directions(restrict_factors([factors], selections), selections)
    mask = torch.tensor([True, False, True])
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="evaluation failure"):
        with perturb(model, bank, mask, 0.03) as changed:
            assert changed["actual_quantized_update_l2"] > 0
            assert any(not torch.equal(state[k], v) for k, v in model.state_dict().items())
            raise RuntimeError("evaluation failure")
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="overflowing"):
        with perturb(model, bank, mask, 1e10):
            raise AssertionError("overflowing updates must be rejected before entering")
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tiny_qwen_probes_and_adam_direction_scores_run_without_base_updates(dtype):
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(72)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    selections = [
        MLPSelection(
            (f"model.layers.{i}.mlp.up_proj", f"model.layers.{i}.mlp.gate_proj"),
            f"model.layers.{i}.mlp.down_proj",
            (4, 0, 9),
        )
        for i in range(2)
    ]
    names = [name for s in selections for name in (*s.incoming, s.outgoing)]
    target = example([1, 4, 8, 6, 2], 3)
    captures = [capture_response_factors(model, target, names)[0]]
    bank = build_directions(restrict_factors(captures, selections), selections)
    probes = capture_fisher_factors(
        model, example([1, 5, 7, 9, 2], 3), selections, probes=2, seed=43
    )
    responses = torch.stack([probe_dots(bank, p) for p in probes])
    result = rank_scores(bank, responses)
    assert result["risk_informative"] and result["scores"].shape == (6,)
    assert torch.isfinite(result["scores"]).all() and (bank.gain > 0).all()
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks for m in model.modules())
