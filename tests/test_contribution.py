import pytest
import torch

from neuronpatch.contribution import contribution_indices


def test_output_direction_not_activation_alone_and_layer_exclusion():
    # Large activation points away from the token. Excluded layer wins otherwise.
    activations = torch.tensor(
        [[[10.0, 2.0, 1.0], [1.0, 1.0, 1.0]], [[10.0, 2.0, 1.0], [1.0, 1.0, 1.0]]],
        requires_grad=True,
    )
    down = [
        torch.tensor([[-1.0, 1.0, 3.0]], requires_grad=True),
        torch.tensor([[100.0, 100.0, 100.0]]),
    ]
    unembedding = torch.tensor([[1.0], [-1.0]], requires_grad=True)
    result = contribution_indices(
        activations, down, unembedding, torch.tensor([0, 1]), 2, exclude_last_layers=1
    )
    assert result.tolist() == [[[0, 2], [0, 1]], [[0, 0], [0, 1]]]
    assert not result.requires_grad
    assert activations.grad is None and down[0].grad is None


def test_repeated_positions_preserve_selection_occurrences():
    result = contribution_indices(
        torch.ones(2, 1, 2),
        [torch.tensor([[1.0, 3.0]])],
        torch.ones(1, 1),
        torch.tensor([0, 0]),
        1,
        exclude_last_layers=0,
    )
    assert result.tolist() == [[[0, 1]], [[0, 1]]]


@pytest.mark.parametrize("change", ["empty", "token", "layers", "k", "overflow"])
def test_invalid_or_nonfinite_scores_rejected(change):
    activation = torch.ones(1, 2, 2)
    weights = [torch.ones(1, 2), torch.ones(1, 2)]
    embedding = torch.ones(2, 1)
    tokens = torch.tensor([0])
    k, excluded = 1, 1
    if change == "empty":
        activation = activation[:0]
    if change == "token":
        tokens = torch.tensor([2])
    if change == "layers":
        excluded = 2
    if change == "k":
        k = 3
    if change == "overflow":
        weights[0].fill_(1e10)
    with pytest.raises(ValueError):
        contribution_indices(
            activation, weights, embedding, tokens, k, exclude_last_layers=excluded
        )
