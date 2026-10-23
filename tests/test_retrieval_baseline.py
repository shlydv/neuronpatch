"""The baseline must preserve weights and record actual generation, not scores."""

import hashlib
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [1, 3, 4]

    def decode(self, tokens, **kwargs):
        return " ".join(str(x) for x in tokens if x != self.eos_token_id)


def test_real_tiny_model_generation_is_recorded_without_updates(tmp_path):
    from safetensors.torch import load_file

    from experiments.retrieval_evidence.run import execute

    torch.manual_seed(43)
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
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows = [{"id": "test", "messages": [], "prompt_tokens": 3}]
    inputs_path = tmp_path / "frozen_inputs.json"
    inputs_path.write_text(json.dumps(rows))
    spec = {
        "runtime_limit_seconds": 60,
        "max_new_tokens": 3,
        "maximum_total_tokens": 6,
        "queries": 1,
        "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
    }
    protocol_path = tmp_path / "frozen_protocol.json"
    protocol_path.write_text(json.dumps(spec))
    execute(model, Tokenizer(), rows, spec, tmp_path / "result")
    outputs = json.loads((tmp_path / "result/predictions.json").read_text())
    first = load_file(str(tmp_path / "result/first_logits.safetensors"))["test"]
    assert outputs[0]["tokens"][0] == first.argmax().item()
    assert outputs[0]["text"] == Tokenizer().decode(outputs[0]["tokens"])
    assert outputs[0]["correctness"] == "UNREVIEWED"
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    from experiments.retrieval_evidence.verify import verify

    checked = verify(tmp_path / "result", inputs_path, protocol_path, Tokenizer())
    assert checked["queries"] == 1 and not checked["first_token_disagreements"]
    outputs[0]["text"] = "tampered"
    (tmp_path / "result/predictions.json").write_text(json.dumps(outputs))
    with pytest.raises(AssertionError):
        verify(tmp_path / "result", inputs_path, protocol_path, Tokenizer())
    with pytest.raises(AssertionError):
        execute(
            model, Tokenizer(), rows, {**spec, "maximum_total_tokens": 5}, tmp_path / "too_long"
        )
