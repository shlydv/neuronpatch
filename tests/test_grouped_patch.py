import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from neuronpatch import NeuronPatch
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


def test_grouped_artifact_roundtrip_merge_and_training_preserve_sharing(tmp_path):
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 3)).double()
    original = deepcopy(model)
    patch = NeuronPatch(model, {"0": [5, 1, 3]}, axis="output", groups={"0": [1, 0, 1]})
    assert patch.num_trainable_parameters == 8
    x = torch.randn(3, 4, dtype=torch.double)
    optimizer = torch.optim.SGD(patch.trainable_parameters(), lr=0.1)
    model(x).square().mean().backward()
    optimizer.step()
    patch.save(tmp_path / "shared")
    loaded = NeuronPatch.load(deepcopy(original), tmp_path / "shared")
    assert loaded.groups == {"0": (1, 0, 1)}
    assert loaded.num_trainable_parameters == 8
    torch.testing.assert_close(model(x), loaded.model(x), rtol=0, atol=0)
    merged = loaded.merged_model()
    torch.testing.assert_close(merged(x), model(x))
    delta = merged[0].weight - original[0].weight
    torch.testing.assert_close(delta[5], delta[3])
    torch.testing.assert_close(delta[[0, 2, 4]], torch.zeros_like(delta[[0, 2, 4]]), rtol=0, atol=0)
    assert type(merged[0]) is nn.Linear
    assert original[0].weight.grad is None


def test_group_manifest_failure_does_not_mutate_model(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4))
    original = deepcopy(model)
    patch = NeuronPatch(model, {"0": [1, 3]}, axis="output", groups={"0": [0, 0]})
    patch.save(tmp_path / "shared")
    path = tmp_path / "shared/patch.json"
    manifest = json.loads(path.read_text())
    manifest["groups"]["0"] = [0, 2]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contiguous"):
        NeuronPatch.load(original, tmp_path / "shared")
    assert type(original[0]) is nn.Linear and original[0].weight.requires_grad


def test_group_attachment_validates_all_targets_before_mutation_and_rejects_nesting():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    with pytest.raises(ValueError, match="one Python integer"):
        NeuronPatch(
            model, {"0": [1, 3], "1": [0]}, axis="output", groups={"0": [0, 0], "1": [0, 1]}
        )
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="output rows"):
        NeuronPatch(model, {"0": [1]}, groups={"0": [0]})
    NeuronPatch(model, {"0": [1, 3]}, axis="output", groups={"0": [0, 0]})
    with pytest.raises(ValueError, match="unpatched"):
        NeuronPatch(model, {"1": [0]})
