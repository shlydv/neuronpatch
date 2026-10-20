"""Exercise actual causal alignment, FP32 row training and merged exports on CPU."""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from experiments.evidence_pairs.run import (  # noqa: E402
    FP32Rows,
    batch,
    fit,
    primary_losses,
    references,
    restore,
)
from experiments.protected_rows.helpers import Examples  # noqa: E402
from experiments.real_qa.response_loss import response_ce, response_logits  # noqa: E402
from neuronpatch import NeuronPatch  # noqa: E402
from neuronpatch.patch import fingerprint  # noqa: E402


class Tokens:
    eos_token_id = pad_token_id = 2

    def encode(self, text, **kwargs):
        return {"answer": [11, 12], "UNKNOWN": [10], "retained": [13]}[text]


class TinyExamples(Examples):
    def prefix(self, row):
        return row["prefix"]


def setup():
    torch.manual_seed(7)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    examples = TinyExamples(Tokens(), torch.device("cpu"))
    pair = {
        "pair_id": "tiny",
        "answer_token": 11,
        "primary": [
            {"id": "yes", "prefix": [3, 4, 5], "answer": "answer"},
            {"id": "no", "prefix": [3, 6], "answer": "UNKNOWN"},
        ],
        "invariant": [
            {"id": "keep1", "prefix": [3, 4, 7], "answer": "retained"},
            {"id": "keep2", "prefix": [3, 6, 7], "answer": "retained"},
        ],
    }
    return model, examples, pair


def test_left_padding_matches_single_example_causal_ce_and_pair_logits():
    model, examples, pair = setup()
    ce, scores = primary_losses(model, pair, examples, 10)
    separate, first = [], []
    for row in pair["primary"]:
        logits, labels = response_logits(model, examples.inputs(row, training=True))
        separate.append(response_ce(logits, labels))
        first.append(logits[0, 0, 10] - logits[0, 0, 11])
    torch.testing.assert_close(ce, torch.stack(separate).mean(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(scores[0], torch.stack(first), atol=1e-6, rtol=1e-5)
    inputs = batch(examples, pair["primary"], training=True)
    assert inputs["attention_mask"][1, 0] == 0
    assert inputs["labels"][1, 0] == -100


def test_fp32_row_hook_changes_only_declared_rows_and_can_be_removed():
    model, _, _ = setup()
    name = "model.layers.0.mlp.up_proj"
    module = model.get_submodule(name)
    x = torch.randn(2, 3, 32)
    before = module(x).detach()
    editor = FP32Rows(model, {name: [1, 9, 17]})
    with torch.no_grad():
        editor.deltas[name].fill_(0.01)
    actual = module(x)
    expected = before.clone()
    expected[..., [1, 9, 17]] += 0.01 * x.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert editor.deltas[name].grad is not None
    assert module.weight.grad is None
    editor.close()
    torch.testing.assert_close(module(x), before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "family,objective", [("guided", "ce+pair"), ("random", "ce"), ("lora", "ce+pair")]
)
def test_training_exports_changed_weights_and_exactly_restores(tmp_path, family, objective):
    model, examples, pair = setup()
    name = "model.layers.0.mlp.up_proj"
    base_hash = fingerprint(model)
    backup = {name: model.get_submodule(name).weight.detach().clone()}
    refs = references(model, [pair], examples, 10, lambda: None)
    protocol = {
        "training_seed": 3,
        "lora_rank": 1,
        "lora_alpha": 2,
        "trainable_parameters": 96,
        "epochs": 2,
        "primary_ce_weight": 0.5,
        "pair_weight": 0.25,
        "control_ce_weight": 0.5,
        "control_kl_weight": 1.0,
        "gradient_clip": 1.0,
    }
    config = {"family": family, "objective": objective, "lr": 0.01, "name": "tiny"}
    out = tmp_path / "fit"
    fit(model, config, {name: [1, 9, 17]}, [pair], refs, examples, 10, protocol, out, lambda: None)
    merged = model.get_submodule(name).weight.detach().clone()
    assert not torch.equal(merged, backup[name])
    restore(model, backup)
    assert fingerprint(model) == base_hash
    if family != "lora":
        loaded = NeuronPatch.load(model, out / "patch")
        expected = copy.deepcopy(backup[name])
        layer = loaded.layers[name]
        expected.index_add_(0, layer.indices, layer.delta.detach())
        torch.testing.assert_close(expected, merged, atol=0, rtol=0)
