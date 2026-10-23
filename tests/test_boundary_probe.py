import pytest
import torch
from torch import nn

from experiments.instruction_boundary.probe_selection import (
    capture_prompt,
    fit_probe,
    select,
    two_clusters,
)


def test_probe_objective_converges_and_standardized_selection_ignores_units():
    signal = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    x = torch.stack(
        [
            signal,
            2 * signal,
            torch.ones_like(signal),
            torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        ],
        dim=1,
    )
    labels = (signal > 0).double()
    before, rng = x.clone(), torch.random.get_rng_state().clone()
    probe = fit_probe(x, labels)
    assert probe["final_objective"] < probe["initial_objective"]
    assert probe["gradient_max_abs"] < 1e-5
    assert probe["training_accuracy"] == 1
    assert probe["weight"][2] == 0
    result = select(x, probe, count=2)
    assert set(result["indices"]) == {0, 1}
    transformed = x * torch.tensor([5.0, 0.01, 2.0, 3.0]) + 7
    other = fit_probe(transformed, labels)
    torch.testing.assert_close(probe["weight"], other["weight"], atol=1e-6, rtol=1e-5)
    assert select(transformed, other, count=2)["indices"] == result["indices"]
    assert torch.equal(x, before) and torch.equal(torch.random.get_rng_state(), rng)
    assert select(x, probe, count=3, random_seed=71) == select(x, probe, count=3, random_seed=71)


def test_two_profile_clusters_and_degenerate_budget_are_explicit():
    result = two_clusters(torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]))
    groups = result["groups"]
    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[0] != groups[2]
    assert not result["degenerate"] and result["inertia"] < 0.02
    duplicate = two_clusters(torch.ones(4, 6))
    assert duplicate["degenerate"] and set(duplicate["groups"]) == {0, 1}
    assert duplicate["inertia"] == 0
    with pytest.raises(ValueError):
        fit_probe(torch.ones(4, 3), torch.ones(4))
    with pytest.raises(ValueError):
        two_clusters(torch.ones(1, 3))


def test_prompt_capture_masks_padding_and_removes_hooks_on_error():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(1, 1, bias=False)
            self.projection.weight.data.fill_(2)
            self.repeat = False

        def forward(self, input_ids, attention_mask, use_cache):
            value = self.projection(input_ids.float().unsqueeze(-1))
            if self.repeat:
                self.projection(input_ids.float().unsqueeze(-1))
            return value

    model = Toy().eval().requires_grad_(False)
    inputs = {
        "input_ids": torch.tensor([[1, 3, 1000]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    result = capture_prompt(model, inputs, ["projection"])
    assert result["projection"].item() == 6
    assert not model.projection._forward_hooks
    model.repeat = True
    with pytest.raises(ValueError, match="execute once"):
        capture_prompt(model, inputs, ["projection"])
    assert not model.projection._forward_hooks
    assert model.projection.weight.grad is None and not model.training
