from types import SimpleNamespace

import torch

from experiments.guarded_training.step import Budget
from experiments.guarded_training.training import fit_guarded, reference_kl, references


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(7), freeze=True)
        self.head = torch.nn.Linear(7, 7, bias=False)
        torch.nn.init.zeros_(self.head.weight)

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def example(source, answer):
    return SimpleNamespace(
        input_ids=torch.tensor([[1, source, answer]]),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        completion_mask=torch.tensor([[False, False, True]]),
    )


def test_anchor_loss_has_zero_reference_gradient_and_matches_independent_kl():
    reference = torch.tensor([[10.0, 1.0, -2.0], [-3.0, 0.0, 0.3]])
    unchanged = reference.clone().requires_grad_()
    reference_kl(unchanged, reference).backward()
    assert torch.equal(unchanged.grad, torch.zeros_like(unchanged))
    current = reference.double() + torch.tensor([[0.5, -0.2, 2], [-1, 0.3, 0.8]]).double()
    oracle = torch.nn.functional.kl_div(
        current.log_softmax(-1),
        reference.double().log_softmax(-1),
        log_target=True,
        reduction="sum",
    ) / len(current)
    torch.testing.assert_close(reference_kl(current, reference.double()), oracle)


def test_actual_training_learns_disjoint_response_with_unchanged_reference_logits():
    model = TokenModel().eval()
    task, preserved = [example(2, 3)], [example(4, 5)]
    old = references(model, preserved)
    result = fit_guarded(
        model,
        model.head.parameters(),
        task,
        preserved,
        old,
        learning_rate=1.0,
        budget=Budget(1e-8, 1e-8),
        steps=10,
    )
    assert result["stop_reason"] == "training_confidence"
    assert result["final"]["task_nll"] < 0.11
    assert result["final"]["mean_kl"] == 0
    assert torch.equal(references(model, preserved)[0], old[0])


def test_conflicting_task_returns_no_change_when_preservation_cannot_be_satisfied():
    model = TokenModel().eval()
    task, preserved = [example(2, 3)], [example(2, 5)]
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    result = fit_guarded(
        model,
        model.head.parameters(),
        task,
        preserved,
        references(model, preserved),
        learning_rate=1.0,
        budget=Budget(0, 0),
        steps=10,
    )
    assert result["stop_reason"] == "no_accepted_update"
    assert result["accepted_updates"] == 0
    assert result["initial"] == result["final"]
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
