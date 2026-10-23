from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.instruction_boundary.row_gradients import capture_rows
from experiments.paired_mlp.selection import capture_response_factors, projection_scores
from experiments.relation_learning.gradient_agreement import GradientBank, compact_gradients
from neuronpatch.losses import completion_nll
from neuronpatch.supervision import ResponseExample


class CausalToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.up = nn.Linear(4, 7)
        self.gate = nn.Linear(4, 7)
        self.down = nn.Linear(7, 4)
        self.head = nn.Linear(4, 11)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds=None, input_ids=None, attention_mask=None, use_cache=False):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        # Mix earlier prompt positions causally into response predictions.
        x = x.cumsum(dim=1)
        hidden = self.down(F.silu(self.gate(x)) * self.up(x))
        return SimpleNamespace(logits=self.head(hidden.cumsum(dim=1)))


def example(ids, start):
    ids = torch.tensor([ids])
    response = torch.zeros_like(ids, dtype=torch.bool)
    response[:, start:] = True
    return ResponseExample(ids, torch.ones_like(response), response, start)


def test_full_response_factors_and_paired_scores_match_actual_dense_mean_gradients():
    torch.manual_seed(48)
    model = CausalToy().eval()
    reference = deepcopy(model)
    model.requires_grad_(False)
    names = ("up", "gate", "down")
    examples = [example([1, 2, 3, 4, 5], 3), example([2, 6, 3, 5], 2)]
    captured, gradients = [], []
    for sample in examples:
        factors, value = capture_response_factors(model, sample, names)
        captured.append(factors)
        reference.zero_grad(set_to_none=True)
        logits = reference(input_ids=sample.input_ids).logits
        loss = completion_nll(
            logits, sample.input_ids, sample.completion_mask, sample.attention_mask
        )
        assert value == pytest.approx(float(loss.detach()), abs=1e-7)
        loss.backward()
        grads = {name: reference.get_submodule(name).weight.grad.clone() for name in names}
        gradients.append(grads)
        for name, (inputs, derivative) in factors.items():
            torch.testing.assert_close(derivative.T @ inputs, grads[name], atol=1e-7, rtol=1e-5)
    scored = projection_scores(captured, incoming=("up", "gate"), outgoing="down")
    mean = {name: torch.stack([g[name] for g in gradients]).double().mean(0) for name in names}
    expected = (
        mean["up"].square().sum(1) + mean["gate"].square().sum(1) + mean["down"].square().sum(0)
    ).sqrt()
    torch.testing.assert_close(scored["paired_gradient_norm"], expected, atol=1e-7, rtol=1e-5)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(not module._forward_hooks for module in model.modules())
    # Prompt positions matter through causal mixing although their logits have no direct loss.
    assert captured[0]["up"][1][0].abs().max() > 0


def test_equal_opposite_prompt_gradients_cancel_before_ranking():
    a, b = torch.randn(3, 4), torch.randn(3, 7)
    out_a, out_b = torch.randn(3, 7), torch.randn(3, 4)
    first = {"up": (a, b), "down": (out_a, out_b)}
    second = {"up": (a, -b), "down": (out_a, -out_b)}
    result = projection_scores([first, second], incoming=("up",), outgoing="down")
    torch.testing.assert_close(
        result["paired_gradient_norm"], torch.zeros(7).double(), atol=1e-7, rtol=0
    )


def test_factor_failure_removes_hooks_and_leaves_base_untouched():
    model = CausalToy().eval().requires_grad_(False)
    ids = example([1, 2, 3], 1)
    ids.completion_mask.zero_()
    with pytest.raises(ValueError):
        capture_response_factors(model, ids, ("up", "down"))
    assert all(not module._forward_hooks for module in model.modules())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_compact_agreement_matches_dense_gradients_and_finite_difference():
    torch.manual_seed(91)
    reference = CausalToy().double().eval()
    model = deepcopy(reference).requires_grad_(False)
    sample = example([1, 2, 3, 4], 2)
    names, indices = ("up", "gate", "down"), (5, 1)
    factors, _ = capture_response_factors(model, sample, names, loss_scale=1)
    compact = compact_gradients(factors, incoming=names[:2], outgoing="down", indices=indices)

    def loss():
        return completion_nll(
            reference(input_ids=sample.input_ids).logits,
            sample.input_ids,
            sample.completion_mask,
            sample.attention_mask,
        )

    loss().backward()
    expected = torch.cat(
        [
            reference.up.weight.grad[list(indices)],
            reference.gate.weight.grad[list(indices)],
            reference.down.weight.grad[:, list(indices)].T,
        ],
        dim=1,
    )
    torch.testing.assert_close(compact, expected, atol=1e-7, rtol=1e-5)
    streamed = capture_rows(model, sample, {"up": indices, "gate": indices}, loss_scale=1)
    torch.testing.assert_close(streamed["up"], expected[:, :4], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(streamed["gate"], expected[:, 4:8], atol=1e-7, rtol=1e-5)
    assert all(not module._forward_hooks for module in model.modules())
    bank = GradientBank()
    bank.add("calibration-example", compact)
    direction = torch.randn_like(compact)
    predicted = bank.report(direction)["predicted_change"]
    original = deepcopy(reference.state_dict())
    values = []
    epsilon = 1e-5
    for scale in [-epsilon, epsilon]:
        reference.load_state_dict(original)
        with torch.no_grad():
            reference.up.weight[list(indices)] += scale * direction[:, :4]
            reference.gate.weight[list(indices)] += scale * direction[:, 4:8]
            reference.down.weight[:, list(indices)] += scale * direction[:, 8:].T
        values.append(float(loss().detach()))
    assert (values[1] - values[0]) / (2 * epsilon) == pytest.approx(predicted, abs=1e-6)


def test_fisher_magnitude_can_hide_cancellation_and_control_interference():
    train, calibration, control = GradientBank(), GradientBank(), GradientBank()
    train.add("train-a", torch.tensor([[10.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
    train.add("train-b", torch.tensor([[-10.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
    report = train.report()
    assert report["mean_squared_norm"] == [100.0, 1.0, 0.0]
    assert report["coherence"] == [0.0, 1.0, None]
    calibration.add("cal-a", torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]))
    control.add("control-a", torch.tensor([[0.0, 0.0], [-3.0, 0.0], [0.0, 0.0]]))
    delta = torch.tensor([[0.0, 0.0], [-0.1, 0.0], [0.0, 0.0]])
    assert calibration.report(delta)["predicted_change"] == pytest.approx(-0.2)
    assert control.report(delta)["predicted_change"] == pytest.approx(0.3)
    # Coherent training updates can still harm controls. No automatic selector follows.
    with pytest.raises(ValueError):
        train.add("train-a", delta)
    with pytest.raises(ValueError):
        train.report(torch.zeros(1, 1))
    with pytest.raises(ValueError):
        GradientBank(max_scalars=1).add("too-large", delta)
