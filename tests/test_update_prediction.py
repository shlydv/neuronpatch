from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.paired_mlp.training import fit
from experiments.relation_learning.update_prediction import compare, predict_loss, preview


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("kind", [torch.optim.Adam, torch.optim.AdamW])
def test_preview_is_exact_and_does_not_change_live_training(kind):
    p = torch.nn.Parameter(torch.tensor([0.3, -0.7], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([1.2], dtype=torch.float64))
    opt = kind(
        [{"params": [p], "lr": 0.03}, {"params": [q], "lr": 0.07}],
        weight_decay=0.2,
        amsgrad=True,
        foreach=False,
    )
    for step in range(4):
        p.grad = torch.tensor([0.2 + step, -0.8], dtype=p.dtype)
        q.grad = torch.ones_like(q) if step % 2 == 0 else None
        state = deepcopy(opt.state_dict())
        before = [x.detach().clone() for x in (p, q)]
        grads = [None if x.grad is None else x.grad.clone() for x in (p, q)]
        rng = torch.get_rng_state().clone()
        proposal = preview(opt)
        prediction = predict_loss(proposal, p.square().sum() + q.square().sum())
        assert_state_equal(state, opt.state_dict())
        assert_state_equal(before, [x.detach() for x in (p, q)])
        assert_state_equal(grads, [x.grad for x in (p, q)])
        assert torch.equal(rng, torch.get_rng_state())
        opt.step()
        result = compare(proposal, prediction, p.square().sum() + q.square().sum())
        # The exact quadratic remainder is ||delta||^2, an independent oracle.
        assert result["linearization_error"] == pytest.approx(
            sum(float(d.square().sum()) for d in proposal.deltas), abs=1e-14
        )


def test_momentum_can_reverse_fresh_gradient_prediction():
    p = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=0.01, foreach=False)
    for _ in range(10):
        p.grad = -torch.ones_like(p)
        opt.step()
    p.grad = torch.full_like(p, 0.1)
    proposal = preview(opt)
    prediction = predict_loss(proposal, p.sum())
    assert prediction["predicted_change"] > 0  # fresh-gradient descent predicts < 0
    opt.step()
    result = compare(proposal, prediction, p.sum())
    assert result["linearization_error"] == pytest.approx(0, abs=1e-15)


def test_finite_step_can_worsen_loss_despite_negative_prediction():
    p = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=1, foreach=False)
    p.grad = 2 * p.detach()
    proposal = preview(opt)
    prediction = predict_loss(proposal, p.square().sum())
    opt.step()
    result = compare(proposal, prediction, p.square().sum())
    assert result["predicted_change"] < 0 < result["observed_change"]
    assert not result["same_sign"]


def test_reject_over_budget_and_stale_predictions():
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.Adam([p])
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="scalar cap"):
        preview(opt, max_scalars=2)
    proposal = preview(opt)
    with torch.no_grad():
        p.add_(1)
    with pytest.raises(ValueError, match="before applying"):
        predict_loss(proposal, p.sum())
    with pytest.raises(ValueError, match="Actual parameters"):
        proposal.check_applied()


def test_observation_preserves_actual_training_loop_bit_for_bit():
    class TokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(7, 7))

        def forward(self, input_ids, attention_mask, use_cache):
            return SimpleNamespace(logits=self.weight[input_ids])

    def example(source, answer):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, source, answer]]),
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            completion_mask=torch.tensor([[False, False, True]]),
        )

    protocol = {
        "seed": 43,
        "steps": 5,
        "preservation_weight": 1.0,
        "loss_scale": 128,
        "gradient_clip": 1.0,
    }
    models = [TokenModel().eval(), TokenModel().eval()]
    pending, measurements = {}, []

    def before(step, row, optimizer):
        proposal = preview(optimizer)
        # A different response from the training response exposes interference.
        loss = -models[1].weight[2].log_softmax(0)[4]
        pending[step] = proposal, predict_loss(proposal, loss)

    def after(step, row, optimizer):
        proposal, prediction = pending.pop(step)
        loss = -models[1].weight[2].log_softmax(0)[4]
        measurements.append(compare(proposal, prediction, loss))

    histories = []
    for index, model in enumerate(models):
        histories.append(
            fit(
                model,
                model.parameters(),
                [example(2, 3)],
                [example(4, 5)],
                protocol,
                {"id": "parity", "lr": 0.03},
                lambda: None,
                before_update=before if index else None,
                after_update=after if index else None,
            )
        )
    assert histories[0] == histories[1]
    assert torch.equal(models[0].weight, models[1].weight)
    assert len(measurements) == 5 and not pending
    assert all(r["predicted_change"] > 0 and r["observed_change"] > 0 for r in measurements)
