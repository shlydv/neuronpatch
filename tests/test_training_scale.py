"""Exercise actual FP16 overflow and Adam, without model downloads."""

from types import SimpleNamespace

import pytest
import torch

from experiments.paired_mlp.training import fit
from neuronpatch.supervision import ResponseExample


class HalfPath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, **kwargs):
        value = self.weight.half() * 2048
        logits = torch.stack((value * 0, value)).expand(1, input_ids.shape[1], 2)
        return SimpleNamespace(logits=logits)


def run(model, *, scale=128, minimum=1, steps=3):
    ids = torch.tensor([[0, 1]])
    sample = ResponseExample(
        ids, torch.ones_like(ids, dtype=torch.bool), torch.tensor([[False, True]]), 1
    )
    protocol = dict(seed=43, steps=steps, preservation_weight=1, loss_scale=scale, gradient_clip=1)
    if minimum is not None:
        protocol["minimum_loss_scale"] = minimum
    return fit(
        model,
        list(model.parameters()),
        [sample],
        [sample],
        protocol,
        dict(lr=0.0001, id="test"),
        lambda: None,
    )


def test_backoff_retries_without_adam_or_example_advancement(monkeypatch):
    instances = []
    original = torch.optim.Adam

    class ObservedAdam(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0
            instances.append(self)

        def step(self, *args, **kwargs):
            self.calls += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", ObservedAdam)
    adapted, reference = HalfPath().eval(), HalfPath().eval()
    history = run(adapted)
    assert history[0]["loss_scale"] == 32
    assert [r["loss_scale"] for r in history[0]["gradient_retries"]] == [128, 64]
    assert all(r["example_indices"] == [0, 0] for r in history)
    assert instances[0].calls == 3
    assert instances[0].state[adapted.weight]["step"] == 3
    baseline = run(reference, scale=32, minimum=None)
    torch.testing.assert_close(adapted.weight, reference.weight, rtol=0, atol=0)
    for actual, expected in zip(history, baseline, strict=True):
        assert {key: actual[key] for key in expected} == expected


def test_unrecoverable_gradient_is_bounded_and_does_not_update():
    model = HalfPath().eval()
    model.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
    with pytest.raises(FloatingPointError, match="step 1.*loss_scale 32"):
        run(model, minimum=32)
    assert model.weight.item() == 0


def test_static_mode_remains_fail_fast():
    model = HalfPath().eval()
    with pytest.raises(FloatingPointError, match="Nonfinite trainable gradient"):
        run(model, minimum=None)
    assert model.weight.item() == 0
