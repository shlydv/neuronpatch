from copy import deepcopy

import pytest
import torch
from torch import nn

from neuronpatch import ColumnPatch, NeuronPatch, random_indices


def make_model():
    torch.manual_seed(12)
    return nn.Sequential(nn.Linear(5, 9), nn.SiLU(), nn.Linear(9, 3)).double()


@pytest.mark.parametrize("axis,shared", [("input", False), ("output", False), ("output", True)])
def test_nested_patch_round_trip_without_newer_module_replacement_api(
    tmp_path, monkeypatch, axis, shared
):
    # Actual PyTorch2.4 lacks this method; simulate that on newer CI versions too.
    monkeypatch.delattr(nn.Module, "set_submodule", raising=False)

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(3, 4)])
            self.heads = nn.ModuleDict({"score": nn.Linear(4, 2)})

        def forward(self, x):
            return self.heads["score"](self.blocks[0](x).tanh())

    model = Nested().double()
    fresh = deepcopy(model)
    targets = {"blocks.0": [0, 2], "heads.score": [1]}
    groups = {"blocks.0": [0, 0], "heads.score": [0]} if shared else None
    patch = NeuronPatch(model, targets, axis=axis, groups=groups)
    with torch.no_grad():
        for delta in patch.trainable_parameters():
            delta.fill_(0.037)
    x = torch.randn(3, 3, dtype=torch.float64)
    expected = model(x)
    merged = patch.merged_model()
    assert type(merged.blocks[0]) is type(merged.heads["score"]) is nn.Linear
    torch.testing.assert_close(merged(x), expected, atol=1e-12, rtol=1e-12)
    patch.save(tmp_path / "patch")
    loaded = NeuronPatch.load(fresh, tmp_path / "patch")
    torch.testing.assert_close(loaded.model(x), expected, atol=0, rtol=0)


def test_forward_and_gradients_match_dense_reference():
    torch.manual_seed(7)
    base = nn.Linear(7, 4).double()
    patch = ColumnPatch(base, [1, 5])
    with torch.no_grad():
        patch.delta.normal_()
    x = torch.randn(2, 3, 7, dtype=torch.double, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_delta = patch.delta.detach().clone().requires_grad_()
    effective = base.weight.detach().clone().index_add(1, patch.indices, reference_delta)
    actual = patch(x)
    expected = nn.functional.linear(reference_x, effective, base.bias.detach())
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(patch.delta.grad, reference_delta.grad)
    assert base.weight.grad is None
    assert base.bias.grad is None


def test_zero_init_freeze_and_column_locality():
    model = make_model()
    original = deepcopy(model)
    x = torch.randn(16, 5, dtype=torch.double)
    patch = NeuronPatch(model, {"2": [2, 7]})
    torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)
    assert patch.num_trainable_parameters == 6
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 6
    optimizer = torch.optim.AdamW(patch.trainable_parameters(), lr=0.1, weight_decay=0.1)
    model(x).square().mean().backward()
    optimizer.step()
    merged = patch.merged_model()
    for name, tensor in original.state_dict().items():
        if name != "2.weight":
            torch.testing.assert_close(merged.state_dict()[name], tensor, rtol=0, atol=0)
    torch.testing.assert_close(
        merged[2].weight[:, [0, 1, 3, 4, 5, 6, 8]],
        original[2].weight[:, [0, 1, 3, 4, 5, 6, 8]],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(model(x), merged(x))
    torch.testing.assert_close(model[2].base.weight, original[2].weight, rtol=0, atol=0)
    assert not any(isinstance(m, ColumnPatch) for m in merged.modules())


def test_save_reload_and_reject_wrong_base(tmp_path):
    model = make_model()
    base = deepcopy(model)
    patch = NeuronPatch(model, {"0": [0, 4], "2": [1, 5]})
    with torch.no_grad():
        for parameter in patch.trainable_parameters():
            parameter.normal_()
    patch.save(tmp_path / "edit", metadata={"base_revision": "test"})
    loaded = NeuronPatch.load(base, tmp_path / "edit")
    x = torch.randn(2, 5, dtype=torch.double)
    torch.testing.assert_close(loaded.model(x), model(x), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        patch.save(tmp_path / "edit")
    wrong = make_model()
    with torch.no_grad():
        wrong[0].weight.add_(1)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        NeuronPatch.load(wrong, tmp_path / "edit")
    assert type(wrong[0]) is nn.Linear
    assert wrong[0].weight.requires_grad


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [9], [True], [1.0]])
def test_invalid_indices_fail_without_mutating_model(indices):
    model = make_model()
    with pytest.raises(ValueError):
        NeuronPatch(model, {"0": [1], "2": indices})
    assert type(model[0]) is nn.Linear
    assert all(p.requires_grad for p in model.parameters())


def test_shared_module_rejected():
    layer = nn.Linear(4, 4)
    model = nn.Sequential(layer, layer)
    with pytest.raises(ValueError, match="shared"):
        NeuronPatch(model, {"0": [1]})


def test_random_selector_is_reproducible_unique_and_local():
    state = torch.random.get_rng_state().clone()
    a = random_indices(100, 10, seed=3)
    assert a == random_indices(100, 10, seed=3)
    assert len(set(a)) == 10
    assert a != random_indices(100, 10, seed=4)
    assert torch.equal(state, torch.random.get_rng_state())


def test_training_recovers_known_edit_on_held_out_inputs():
    model = make_model()
    teacher = deepcopy(model)
    with torch.no_grad():
        teacher[2].weight[:, 2] += torch.tensor([0.4, -0.2, 0.3])
    train_x = torch.randn(128, 5, dtype=torch.double)
    test_x = torch.randn(128, 5, dtype=torch.double)
    with torch.no_grad():
        train_y, test_y = teacher(train_x), teacher(test_x)
        initial = nn.functional.mse_loss(model(test_x), test_y).item()
    patch = NeuronPatch(model, {"2": [2]})
    optimizer = torch.optim.Adam(patch.trainable_parameters(), lr=0.03)
    for _ in range(150):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    final = nn.functional.mse_loss(model(test_x), test_y).item()
    assert final < initial * 0.001


def test_corrupt_tensor_shape_rejected_before_attachment(tmp_path):
    from safetensors.torch import save_file

    model = make_model()
    base = deepcopy(model)
    patch = NeuronPatch(model, {"2": [2]})
    patch.save(tmp_path / "edit")
    save_file(
        {"2": torch.zeros(1, dtype=torch.double)}, str(tmp_path / "edit" / "patch.safetensors")
    )
    with pytest.raises(ValueError, match="shape"):
        NeuronPatch.load(base, tmp_path / "edit")
    assert type(base[2]) is nn.Linear
