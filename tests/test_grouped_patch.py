from copy import deepcopy

import pytest
import torch
from torch import nn

from neuronpatch.grouped import GroupedRowPatch
from neuronpatch.layers import RowPatch


def test_shared_update_forward_input_gradient_and_summed_group_gradient():
    torch.manual_seed(71)
    base = nn.Linear(5, 7).double()
    original = deepcopy(base)
    patch = GroupedRowPatch(base, [6, 1, 3], [1, 0, 1])
    with torch.no_grad():
        patch.delta.normal_()
    x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    # Independent row deltas provide a separate reference for summed gradients.
    rows = patch.delta.detach().index_select(0, patch.groups).clone().requires_grad_()
    effective = original.weight.detach().index_add(0, patch.indices, rows)
    expected = nn.functional.linear(rx, effective, original.bias.detach())
    actual = patch(x)
    torch.testing.assert_close(actual, expected)
    coefficients = torch.randn_like(actual)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    torch.testing.assert_close(patch.delta.grad[0], rows.grad[1])
    torch.testing.assert_close(patch.delta.grad[1], rows.grad[0] + rows.grad[2])
    assert base.weight.grad is None and base.bias.grad is None
    merged = patch.merged_linear()
    torch.testing.assert_close(merged(x), actual)
    torch.testing.assert_close(
        merged.weight[[0, 2, 4, 5]], original.weight[[0, 2, 4, 5]], rtol=0, atol=0
    )
    torch.testing.assert_close(base.weight, original.weight, rtol=0, atol=0)
    assert merged.weight.data_ptr() != base.weight.data_ptr()
    assert patch.delta.numel() == 10


def test_singleton_groups_match_independent_rows_including_gradients():
    base = nn.Linear(4, 6, bias=False).double()
    grouped = GroupedRowPatch(deepcopy(base), [4, 0, 2], [0, 1, 2])
    independent = RowPatch(deepcopy(base), [4, 0, 2])
    x = torch.randn(3, 4, dtype=torch.double)
    torch.testing.assert_close(grouped(x), base(x), rtol=0, atol=0)
    with torch.no_grad():
        grouped.delta.normal_()
        independent.delta.copy_(grouped.delta)
    torch.testing.assert_close(grouped(x), independent(x))
    grouped(x).square().sum().backward()
    independent(x).square().sum().backward()
    torch.testing.assert_close(grouped.delta.grad, independent.delta.grad)


@pytest.mark.parametrize(
    "indices,groups", [([0, 0], [0, 1]), ([0, 1], [0]), ([0, 1], [0, 2]), ([0, 1], [False, 0])]
)
def test_invalid_grouping_does_not_freeze_the_callers_layer(indices, groups):
    base = nn.Linear(4, 6)
    with pytest.raises(ValueError):
        GroupedRowPatch(base, indices, groups)
    assert base.weight.requires_grad and base.bias.requires_grad
