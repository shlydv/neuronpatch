import pytest
import torch
from torch import nn

from neuronpatch import contrastive_indices, mean_activations


class GatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(3, 2)

    def forward(self, x, attention_mask=None):
        return self.down_proj(x * torch.sigmoid(x))


def test_collects_gated_inputs_with_equal_example_weight():
    model = GatedModel()
    model.train()
    model.down_proj.eval()
    x = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[True, True], [True, False]])
    result = mean_activations(model, ["down_proj"], [({"x": x}, mask)])
    gated = x * x.sigmoid()
    expected = (gated[0].double().mean(0) + gated[1, 0].double()) / 2
    torch.testing.assert_close(result["down_proj"], expected)
    assert model.training and not model.down_proj.training
    assert not model.down_proj._forward_pre_hooks
    split = [({"x": x[i : i + 1]}, mask[i : i + 1]) for i in range(2)]
    torch.testing.assert_close(mean_activations(model, ["down_proj"], split)["down_proj"], expected)


def test_hook_cleanup_on_forward_failure():
    model = GatedModel().train()
    with pytest.raises(ValueError, match="must match"):
        mean_activations(
            model, ["down_proj"], [({"x": torch.ones(1, 3)}, torch.ones(1, 1, dtype=torch.bool))]
        )
    assert model.training
    assert not model.down_proj._forward_pre_hooks


def test_padding_and_empty_examples_rejected():
    model = GatedModel()
    with pytest.raises(ValueError, match="padding"):
        mean_activations(
            model,
            ["down_proj"],
            [
                (
                    {"x": torch.ones(1, 2, 3), "attention_mask": torch.tensor([[1, 0]])},
                    torch.tensor([[True, True]]),
                )
            ],
        )
    with pytest.raises(ValueError, match="at least one"):
        mean_activations(
            model,
            ["down_proj"],
            [
                (
                    {"x": torch.ones(1, 2, 3)},
                    torch.zeros(1, 2, dtype=torch.bool),
                )
            ],
        )
    with pytest.raises(ValueError, match="empty"):
        mean_activations(model, ["down_proj"], [])


def test_contrast_ranking_ties_and_validation():
    positive = torch.tensor([1.0, -5.0, 5.0, 0.0])
    negative = torch.zeros(4)
    assert contrastive_indices(positive, negative, 1) == [1]
    assert contrastive_indices(positive, negative, 2) == [1, 2]
    with pytest.raises(ValueError, match="finite"):
        contrastive_indices(torch.tensor([float("nan")]), torch.zeros(1), 1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple GPU")
def test_mps_statistics_match_cpu():
    # Transfer first, then cast: a combined MPS->CPU float64 conversion is unsafe
    # on some torch/MPS versions, and float64 cannot be computed on MPS itself.
    from copy import deepcopy

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.proj(x)

    cpu = Model()
    gpu = deepcopy(cpu).to("mps")
    x = torch.tensor([[[1.25, -3.5, 8.0], [2.5, 7.0, -1.0]]])
    mask = torch.tensor([[True, False]])
    expected = mean_activations(cpu, ["proj"], [({"x": x}, mask)])
    actual = mean_activations(gpu, ["proj"], [({"x": x.to("mps")}, mask.to("mps"))])
    torch.testing.assert_close(actual["proj"], expected["proj"], atol=0, rtol=0)
