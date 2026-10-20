from copy import deepcopy

import pytest
import torch
from torch import nn

from neuronpatch.occurrence import OccurrenceColumnEdit


def test_repeated_columns_and_layers_match_dense_merge_and_keep_base_frozen():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4)).double()
    original = deepcopy(model.state_dict())
    edit = OccurrenceColumnEdit(model, [("0", 2), ("2", 1), ("0", 2)])
    with torch.no_grad():
        edit.delta[0].fill_(0.125)
        edit.delta[1].fill_(-0.25)
        edit.delta[2].fill_(0.25)
    x = torch.randn(1, 3, 4, dtype=torch.float64)
    expected = deepcopy(model)
    with torch.no_grad():
        expected[0].weight[:, 2] += 0.375
        expected[2].weight[:, 1] -= 0.25
    torch.testing.assert_close(edit(x), expected(x), atol=1e-14, rtol=1e-14)
    merged = edit.merged_model()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(merged.state_dict()[name], tensor, atol=0, rtol=0)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert edit.num_trainable_parameters == 12
    assert not any(p.requires_grad for p in model.parameters())
    assert list(edit.trainable_parameters()) == [edit.delta]
    assert not model[0]._forward_hooks and not model[2]._forward_hooks


def test_context_cleanup_on_error_and_nested_context_rejection():
    model = nn.Sequential(nn.Linear(2, 2))
    edit = OccurrenceColumnEdit(model, [("0", 0)])
    with pytest.raises(RuntimeError, match="test failure"):
        with edit.applied():
            with pytest.raises(RuntimeError, match="already active"):
                with edit.applied():
                    pass
            with pytest.raises(RuntimeError, match="before merging"):
                edit.merged_model()
            raise RuntimeError("test failure")
    assert not edit._active and not model[0]._forward_hooks
    edit(torch.ones(1, 1, 2))


def test_validation_happens_before_freezing():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
    for targets in ([("0", 0), ("absent", 0)], [("0", 0), ("1", 0)], [("0", True)]):
        with pytest.raises(ValueError):
            OccurrenceColumnEdit(model, targets)
        assert all(p.requires_grad for p in model.parameters())


def test_joint_norm_has_independent_occurrence_gradients():
    model = nn.Sequential(nn.Linear(2, 2)).double()
    edit = OccurrenceColumnEdit(model, [("0", 0), ("0", 0)])
    with torch.no_grad():
        edit.delta.copy_(torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64))
    assert edit.delta_norm().item() == 5
    edit.delta_norm().backward()
    torch.testing.assert_close(edit.delta.grad, edit.delta.detach() / 5)
