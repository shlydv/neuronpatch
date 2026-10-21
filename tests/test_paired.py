from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from neuronpatch import NeuronPatch
from neuronpatch.layers import ColumnPatch, RowPatch
from neuronpatch.paired import MLPSelection, PairedMLPPatch
from neuronpatch.patch import fingerprint


class MLP(nn.Module):
    def __init__(self, gated):
        super().__init__()
        self.up = nn.Linear(4, 7)
        if gated:
            self.gate = nn.Linear(4, 7)
        self.down = nn.Linear(7, 4)

    def forward(self, x):
        features = self.up(x)
        features = F.silu(self.gate(x)) * features if hasattr(self, "gate") else F.gelu(features)
        return self.down(features)


def selection(gated):
    return MLPSelection(("up", "gate") if gated else ("up",), "down", (1, 5))


@pytest.mark.parametrize("gated", [False, True])
def test_paired_nonlinear_forward_gradients_merge_and_reload_match_dense(gated, tmp_path):
    torch.manual_seed(27)
    model = MLP(gated).double()
    original = deepcopy(model)
    base_id = fingerprint(model)
    patch = PairedMLPPatch(model, [selection(gated)], delta_dtype=torch.float64)
    assert patch.base_fingerprint == base_id
    assert patch.num_trainable_parameters == (3 if gated else 2) * 2 * 4
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model(x), original(x), atol=0, rtol=0)
    with torch.no_grad():
        for delta in patch.trainable_parameters():
            delta.normal_(std=0.15)
    dense_x = x.detach().clone().requires_grad_()
    reference = {
        n: layer.delta.detach().clone().requires_grad_() for n, layer in patch.layers.items()
    }
    incoming = {}
    for name in selection(gated).incoming:
        layer = original.get_submodule(name)
        effective = layer.weight.detach().index_add(0, torch.tensor([1, 5]), reference[name])
        incoming[name] = F.linear(dense_x, effective, layer.bias.detach())
    features = F.silu(incoming["gate"]) * incoming["up"] if gated else F.gelu(incoming["up"])
    down = original.down.weight.detach().index_add(1, torch.tensor([1, 5]), reference["down"])
    expected = F.linear(features, down, original.down.bias.detach())
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, dense_x.grad, atol=1e-12, rtol=1e-12)
    for name, layer in patch.layers.items():
        torch.testing.assert_close(layer.delta.grad, reference[name].grad, atol=1e-12, rtol=1e-12)
        assert layer.base.weight.grad is None
        assert layer.base.bias.grad is None
    merged = patch.merged_model()
    assert not any(isinstance(m, (ColumnPatch, RowPatch)) for m in merged.modules())
    torch.testing.assert_close(merged(x), actual, atol=1e-12, rtol=1e-12)
    for name, layer in patch.layers.items():
        weight = merged.get_submodule(name).weight
        old = original.get_submodule(name).weight
        if name == "down":
            torch.testing.assert_close(
                weight[:, [0, 2, 3, 4, 6]], old[:, [0, 2, 3, 4, 6]], rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(
                weight[[0, 2, 3, 4, 6]], old[[0, 2, 3, 4, 6]], rtol=0, atol=0
            )
    patch.save(tmp_path / "paired", metadata={"test": "dense nonlinear reference"})
    loaded = PairedMLPPatch.load(deepcopy(original), tmp_path / "paired")
    torch.testing.assert_close(loaded.model(x), actual, atol=0, rtol=0)
    torch.testing.assert_close(loaded.merged_model()(x), merged(x), atol=0, rtol=0)
    assert loaded.selections == patch.selections
    assert all(
        not p.requires_grad for layer in patch.layers.values() for p in layer.base.parameters()
    )


def test_half_base_fp32_updates_autocast_and_promoted_merge(tmp_path):
    torch.manual_seed(31)
    model = MLP(True).half()
    fresh = deepcopy(model)
    patch = PairedMLPPatch(model, [selection(True)])
    x = torch.randn(5, 4).half()
    torch.testing.assert_close(model(x), fresh(x), atol=0, rtol=0)
    for layer in patch.layers.values():
        assert layer.delta.dtype == torch.float32
        with torch.no_grad():
            layer.delta.normal_(std=0.01)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(x)
        loss = result.float().square().mean()
    assert result.dtype == torch.bfloat16
    loss.backward()
    for parameter in patch.trainable_parameters():
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0
    optimizer = torch.optim.Adam(patch.trainable_parameters(), lr=0.001)
    optimizer.step()
    merged = patch.merged_model()
    for name, layer in patch.layers.items():
        expected = fresh.get_submodule(name).weight.float().clone()
        expected.index_add_(1 if name == "down" else 0, torch.tensor([1, 5]), layer.delta.detach())
        torch.testing.assert_close(
            merged.get_submodule(name).weight, expected.half(), atol=0, rtol=0
        )
    patch.save(tmp_path / "half")
    restored = PairedMLPPatch.load(fresh, tmp_path / "half")
    torch.testing.assert_close(restored.model(x), model(x), atol=0, rtol=0)


@pytest.mark.parametrize("problem", ["width", "duplicate", "index", "tied", "dtype"])
def test_invalid_request_does_not_mutate_base(problem):
    model = MLP(True)
    picks = [selection(True)]
    kwargs = {}
    if problem == "width":
        model.gate = nn.Linear(4, 8)
    elif problem == "duplicate":
        picks.append(selection(True))
    elif problem == "index":
        picks = [MLPSelection(("up", "gate"), "down", (7,))]
    elif problem == "tied":
        model.gate.weight = model.up.weight
    else:
        kwargs["delta_dtype"] = torch.float16
    original = fingerprint(model)
    with pytest.raises(ValueError):
        PairedMLPPatch(model, picks, **kwargs)
    assert fingerprint(model) == original
    assert all(p.requires_grad for p in model.parameters())
    assert type(model.up) is nn.Linear


def test_nested_attachment_wrong_base_and_corruption_rejected(tmp_path):
    model = MLP(False)
    fresh = deepcopy(model)
    patch = PairedMLPPatch(model, [selection(False)])
    with pytest.raises(ValueError, match="unpatched"):
        PairedMLPPatch(model, [selection(False)])
    with pytest.raises(ValueError, match="unpatched"):
        NeuronPatch(model, {"up.base": [1]})
    patch.save(tmp_path / "edit")
    wrong = deepcopy(fresh)
    with torch.no_grad():
        wrong.down.weight.add_(0.01)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        PairedMLPPatch.load(wrong, tmp_path / "edit")
    assert type(wrong.up) is nn.Linear and wrong.up.weight.requires_grad
    path = tmp_path / "edit" / "patch.safetensors"
    tensors = load_file(str(path))
    tensors["down"][0, 0] = float("nan")
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="invalid paired delta"):
        PairedMLPPatch.load(fresh, tmp_path / "edit")
    assert type(fresh.up) is nn.Linear and fresh.up.weight.requires_grad
    with torch.no_grad():
        model.up.base.bias.add_(0.01)
    with pytest.raises(ValueError, match="base model changed"):
        patch.merged_model()
    with pytest.raises(ValueError, match="base model changed"):
        patch.save(tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_changed_indices_cannot_be_saved_under_original_manifest(tmp_path):
    patch = PairedMLPPatch(MLP(False), [selection(False)])
    patch.model.up.indices[0] = 2
    with pytest.raises(ValueError, match="patch indices"):
        patch.save(tmp_path / "wrong-selection")
    assert not (tmp_path / "wrong-selection").exists()


def test_gated_qwen_full_response_step_reload_and_merge(tmp_path):
    transformers = pytest.importorskip("transformers")
    from experiments.paired_mlp.selection import capture_response_factors, projection_scores
    from neuronpatch.losses import completion_nll
    from neuronpatch.supervision import ResponseExample

    torch.manual_seed(44)
    config = transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).eval()
    original = deepcopy(model)
    picks = [
        MLPSelection(
            ("model.layers.1.mlp.gate_proj", "model.layers.1.mlp.up_proj"),
            "model.layers.1.mlp.down_proj",
            (2, 7),
        )
    ]
    ids = torch.tensor([[1, 2, 3, 4, 2]])
    attention = torch.ones_like(ids, dtype=torch.bool)
    response = torch.tensor([[False, False, False, True, True]])
    model.requires_grad_(False)
    captured, _ = capture_response_factors(
        model,
        ResponseExample(ids, attention, response, 3),
        (*picks[0].incoming, picks[0].outgoing),
    )
    scores = projection_scores([captured], incoming=picks[0].incoming, outgoing=picks[0].outgoing)
    assert scores["paired_gradient_norm"].shape == (32,)
    assert scores["paired_gradient_norm"].max() > 0
    patch = PairedMLPPatch(model, picks)
    optimizer = torch.optim.Adam(patch.trainable_parameters(), lr=0.01)
    logits = model(input_ids=ids, attention_mask=attention).logits
    completion_nll(logits, ids, response, attention).backward()
    assert patch.num_trainable_parameters == 96
    for parameter in patch.trainable_parameters():
        assert parameter.grad.abs().max() > 0
    optimizer.step()
    expected = model(input_ids=ids, attention_mask=attention).logits
    patch.save(tmp_path / "qwen")
    restored = PairedMLPPatch.load(original, tmp_path / "qwen")
    torch.testing.assert_close(
        restored.model(input_ids=ids, attention_mask=attention).logits, expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        patch.merged_model()(input_ids=ids, attention_mask=attention).logits,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )
