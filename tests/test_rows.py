from copy import deepcopy

import pytest
import torch
from torch import nn

from neuronpatch import NeuronPatch
from neuronpatch.layers import RowPatch


def test_row_update_matches_dense_gradients_and_freezes_other_rows(tmp_path):
    torch.manual_seed(15)
    model = nn.Sequential(nn.Linear(3, 5)).double()
    original = deepcopy(model)
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    patch = NeuronPatch(model, {"0": [1, 4]}, axis="output")
    torch.testing.assert_close(model(x), original(x), atol=0, rtol=0)
    layer = model[0]
    with torch.no_grad():
        layer.delta.normal_()
    full_delta = torch.zeros_like(original[0].weight)
    full_delta[[1, 4]] = layer.delta.detach()
    full_delta.requires_grad_()
    expected = torch.nn.functional.linear(x, original[0].weight + full_delta, original[0].bias)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    dx, dd = torch.autograd.grad(actual.square().sum(), (x, layer.delta))
    ex, ed = torch.autograd.grad(expected.square().sum(), (x, full_delta))
    torch.testing.assert_close(dx, ex)
    torch.testing.assert_close(dd, ed[[1, 4]])
    optimizer = torch.optim.AdamW(patch.trainable_parameters(), lr=0.01)
    model(x).square().sum().backward()
    optimizer.step()
    merged = patch.merged_model()
    torch.testing.assert_close(model(x), merged(x))
    torch.testing.assert_close(
        merged[0].weight[[0, 2, 3]], original[0].weight[[0, 2, 3]], atol=0, rtol=0
    )
    torch.testing.assert_close(model[0].base.weight, original[0].weight, atol=0, rtol=0)
    torch.testing.assert_close(merged[0].bias, original[0].bias, atol=0, rtol=0)
    patch.save(tmp_path / "rows")
    restored = NeuronPatch.load(deepcopy(original), tmp_path / "rows")
    assert isinstance(restored.model[0], RowPatch)
    torch.testing.assert_close(restored.model(x), model(x), atol=0, rtol=0)


def test_row_validation_uses_output_width_and_is_atomic():
    model = nn.Sequential(nn.Linear(3, 5))
    with pytest.raises(ValueError):
        NeuronPatch(model, {"0": [5]}, axis="output")
    assert type(model[0]) is nn.Linear
    assert model[0].weight.requires_grad
    with pytest.raises(ValueError):
        NeuronPatch(model, {"0": [1]}, axis="rows")
    NeuronPatch(model, {"0": [4]}, axis="output")
    with pytest.raises(ValueError):
        NeuronPatch(model, {"0.base": [1]})


def test_gate_rows_on_tiny_qwen_train_and_merge():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).eval()
    patch = NeuronPatch(model, {"model.layers.0.mlp.gate_proj": [2, 7]}, axis="output")
    ids = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.Adam(patch.trainable_parameters(), lr=0.01)
    model(input_ids=ids, labels=ids).loss.backward()
    assert torch.count_nonzero(next(patch.trainable_parameters()).grad) > 0
    optimizer.step()
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=ids).logits,
            patch.merged_model()(input_ids=ids).logits,
            atol=1e-6,
            rtol=1e-5,
        )
