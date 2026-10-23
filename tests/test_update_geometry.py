import pytest
import torch

from experiments.relation_learning.update_geometry import geometry


def test_identical_and_opposing_updates_distinguish_sharing_from_rank():
    v = torch.tensor([1.0, 2.0, -3.0])
    same = geometry(torch.stack([v, v]))
    opposite = geometry(torch.stack([v, -v]))
    assert same["identical_row_fraction"] == pytest.approx(1)
    assert same["rank_one_fraction"] == pytest.approx(1)
    assert opposite["identical_row_fraction"] == pytest.approx(0)
    assert opposite["rank_one_fraction"] == pytest.approx(1)


def test_orthogonal_and_zero_updates():
    result = geometry(torch.eye(4))
    assert result["identical_row_fraction"] == pytest.approx(0.25)
    assert result["rank_one_fraction"] == pytest.approx(0.25)
    assert geometry(torch.zeros(3, 5))["rank_one_fraction"] is None
    with pytest.raises(ValueError):
        geometry(torch.tensor([[float("nan")]]))


def test_energy_matches_independent_svd_reconstruction():
    matrix = torch.randn(5, 13, generator=torch.Generator().manual_seed(43)).double()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    reconstruction = s[0] * u[:, :1] @ vh[:1]
    result = geometry(matrix)
    error = (matrix - reconstruction).square().sum().item()
    assert result["energy"] - result["rank_one_energy"] == pytest.approx(error)
    shared_error = (matrix - matrix.mean(0, keepdim=True)).square().sum().item()
    assert result["energy"] - result["identical_row_energy"] == pytest.approx(shared_error)
