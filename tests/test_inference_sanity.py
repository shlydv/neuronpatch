"""Check the diagnostic's generation and canonical identity on a tiny real model."""

import pytest
import torch

transformers = pytest.importorskip("transformers")


def test_diagnostic_records_outputs_and_identifies_same_rounded_weights():
    from experiments.inference_sanity.run import canonical_fp16_fingerprint, generate
    from neuronpatch.patch import fingerprint
    from tests.test_retrieval_baseline import Tokenizer

    torch.manual_seed(9)
    model = (
        transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                eos_token_id=2,
            )
        )
        .eval()
        .requires_grad_(False)
        .half()
    )
    expected = fingerprint(model)
    assert canonical_fp16_fingerprint(model) == expected
    model.float()
    assert canonical_fp16_fingerprint(model) == expected
    before = fingerprint(model)
    logits = {}
    output = generate(
        model, Tokenizer(), [{"id": "one", "messages": [], "prompt_tokens": 3}], 4, "fp32", logits
    )
    assert output[0]["tokens"][0] == logits["fp32:one"].argmax().item()
    assert output[0]["text"] == Tokenizer().decode(output[0]["tokens"])
    assert fingerprint(model) == before
