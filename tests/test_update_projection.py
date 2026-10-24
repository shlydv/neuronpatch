import pytest
import torch

from experiments.instruction_boundary.update_projection import energies


def test_matches_explicit_dense_projection_and_orthogonal_residual():
    generator = torch.Generator().manual_seed(41)
    a = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    b = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    indices, groups = [8, 1, 5, 3], [0, 1, 0, 1]
    dense = b @ a
    projected = torch.zeros_like(dense)
    for label in set(groups):
        members = [row for row, group in zip(indices, groups) if group == label]
        projected[members] = dense[members].mean(0)
    result = energies(a, b, indices, groups)
    assert result["full"] == pytest.approx(float(dense.square().sum()))
    assert result["selected_shared"] == pytest.approx(float(projected.square().sum()))
    assert result["selected_independent"] == pytest.approx(float(dense[indices].square().sum()))
    assert float((projected * (dense - projected)).sum()) == pytest.approx(0, abs=1e-12)
    assert float((dense - projected).square().sum()) == pytest.approx(
        result["full"] - result["selected_shared"]
    )
    assert result["selected_shared"] <= result["selected_independent"]
    assert result["selected_independent"] <= result["oracle_top_rows_independent"]


def test_opposite_rows_cancel_under_sharing_but_not_independent_rows():
    result = energies(torch.tensor([[2.0, 3.0]]), torch.tensor([[1.0], [-1.0]]), [0, 1], [0, 0])
    assert result["selected_shared"] == 0
    assert result["selected_independent"] == 26
    independent = energies(
        torch.tensor([[2.0, 3.0]]), torch.tensor([[1.0], [-1.0]]), [0, 1], [0, 1]
    )
    assert independent["selected_shared"] == 26
