"""Full-model gradient oracles for marginal response supervision."""

import copy

import pytest
import torch
from test_paired_worker import TinyTokenizer

from experiments.alias_supervision.loss import canonical_loss, finite_response_mean_nll
from neuronpatch.supervision import chat_response


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_multi_alias_loss_and_parameter_gradients_match_independent_full_graph(dtype):
    tf = pytest.importorskip("transformers")
    config = tf.Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(19)
    model = tf.Qwen3ForCausalLM(config).to(dtype).eval()
    other = copy.deepcopy(model)
    tokenizer = TinyTokenizer()
    messages = [{"role": "user", "content": "answer"}]
    samples = [
        chat_response(tokenizer, messages, text, max_length=32)
        for text in ["red blue", "red", "green", "red"]
    ]
    actual = finite_response_mean_nll(model, samples, eos_token_id=2)
    # Independent construction of one conditional vector per shared prefix.
    prefix_logits = {}
    paths = []
    for sample in samples:
        path = tuple(sample.input_ids[0, sample.prompt_length :].tolist())
        if path not in paths:
            paths.append(path)
        z = other(
            input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
        ).logits
        for i in range(len(path)):
            prefix_logits.setdefault(path[:i], z[0, sample.prompt_length + i - 1].double())
    sequence_logs = []
    for path in paths:
        terms = [
            prefix_logits[path[:i]][t] - prefix_logits[path[:i]].logsumexp(0)
            for i, t in enumerate(path)
        ]
        sequence_logs.append(sum(terms))
    expected = -torch.logsumexp(torch.stack(sequence_logs), 0) / len(paths[0])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    (actual * 128).backward()
    (expected * 128).backward()
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_singleton_loss_uses_exact_existing_canonical_path():
    tf = pytest.importorskip("transformers")
    config = tf.Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = tf.Qwen3ForCausalLM(config).eval()
    tokenizer = TinyTokenizer()
    sample = chat_response(tokenizer, [{"role": "user", "content": "answer"}], "red", max_length=16)
    a = finite_response_mean_nll(model, [sample, sample], eos_token_id=2)
    b = canonical_loss(model, sample)
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a, tuple(model.parameters()))
    gb = torch.autograd.grad(b, tuple(model.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(ga, gb, strict=True))


def test_different_conditioning_prompts_are_rejected():
    from torch import nn

    tokenizer = TinyTokenizer()
    samples = [
        chat_response(tokenizer, [{"role": "user", "content": p}], "red", max_length=16)
        for p in ["first", "second"]
    ]
    with pytest.raises(ValueError, match="identical nonempty prompt"):
        finite_response_mean_nll(nn.Linear(1, 1).eval(), samples, eos_token_id=2)
