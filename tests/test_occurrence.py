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


def test_compact_checkpoint_preserves_duplicates_precision_and_outputs(tmp_path):
    import json

    from safetensors.torch import load_file

    torch.manual_seed(5)
    base = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4)).half().eval()
    replica = deepcopy(base)
    edit = OccurrenceColumnEdit(base, [("0", 2), ("2", 1), ("0", 2)], delta_dtype=torch.bfloat16)
    with torch.no_grad():
        edit.delta.normal_(std=0.03)
    path = tmp_path / "edit"
    edit.save(path, metadata={"recipe": "test"})
    tensors = load_file(str(path / "edit.safetensors"))
    assert set(tensors) == {"delta"} and tensors["delta"].shape == (3, 4)
    assert tensors["delta"].dtype == torch.bfloat16
    config = json.loads((path / "edit.json").read_text())
    assert config["targets"] == [["0", 2], ["2", 1], ["0", 2]]
    assert config["metadata"] == {"recipe": "test"}
    loaded = OccurrenceColumnEdit.load(replica, path)
    assert loaded.targets == edit.targets and loaded.delta.dtype == torch.bfloat16
    x = torch.randn(1, 2, 4).half()
    torch.testing.assert_close(loaded(x), edit(x), atol=0, rtol=0)
    for name, tensor in edit.merged_model().state_dict().items():
        torch.testing.assert_close(tensor, loaded.merged_model().state_dict()[name], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        edit.save(path)


@pytest.mark.parametrize(
    "corruption", ["shape", "dtype", "nan", "keys", "target", "format", "base"]
)
def test_bad_checkpoint_rejected_before_mutating_base(tmp_path, corruption):
    import json

    from safetensors.torch import load_file, save_file

    base = nn.Sequential(nn.Linear(3, 3))
    replica = deepcopy(base)
    edit = OccurrenceColumnEdit(base, [("0", 1), ("0", 1)])
    path = tmp_path / "edit"
    edit.save(path)
    if corruption in ("shape", "dtype", "nan", "keys"):
        tensors = load_file(str(path / "edit.safetensors"))
        if corruption == "shape":
            tensors["delta"] = tensors["delta"][:1]
        if corruption == "dtype":
            tensors["delta"] = tensors["delta"].double()
        if corruption == "nan":
            tensors["delta"][0, 0] = float("nan")
        if corruption == "keys":
            tensors["other"] = tensors.pop("delta")
        save_file(tensors, str(path / "edit.safetensors"))
    elif corruption in ("target", "format"):
        config = json.loads((path / "edit.json").read_text())
        if corruption == "target":
            config["targets"][1][1] = True
        if corruption == "format":
            config["format_version"] = True
        (path / "edit.json").write_text(json.dumps(config))
    else:
        with torch.no_grad():
            replica[0].weight[0, 0] = 42
    original = deepcopy(replica.state_dict())
    with pytest.raises(ValueError):
        OccurrenceColumnEdit.load(replica, path)
    assert all(p.requires_grad for p in replica.parameters())
    for name, tensor in replica.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert not replica[0]._forward_hooks


def test_save_rejects_active_context_changed_base_and_nonfinite_delta(tmp_path):
    base = nn.Sequential(nn.Linear(3, 3))
    edit = OccurrenceColumnEdit(base, [("0", 1)])
    with edit.applied():
        with pytest.raises(RuntimeError):
            edit.save(tmp_path / "active")
    with torch.no_grad():
        edit.delta[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        edit.save(tmp_path / "nan")
    with torch.no_grad():
        edit.delta.zero_()
        base[0].weight[0, 0] = 42
    with pytest.raises(ValueError, match="base model changed"):
        edit.save(tmp_path / "changed")
    assert list(tmp_path.iterdir()) == []
