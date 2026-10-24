import json

import torch
from test_paired_worker import TinyTokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.instruction_boundary.preference_run import execute
from neuronpatch.patch import fingerprint


def test_full_half_precision_worker_checkpointing_generation_and_exports(tmp_path):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(31)
    model = Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    original = fingerprint(model)
    tokenizer = TinyTokenizer()
    data = {}
    for role in ("train", "calibration"):
        messages = [{"role": "user", "content": role + " table question"}]
        pair = {"id": role + "-pair", "messages": messages, "chosen": "seven", "rejected": "eight"}
        data[role] = {
            "tasks": [{"document_id": role}],
            "narrow_pairs": [pair],
            "diverse_pairs": [pair],
            "clean": [{"messages": messages, "target": "seven"}],
        }
    data["confirmation"] = {
        "tasks": [{"document_id": "confirmation"}],
        "cases": [{"id": "test", "messages": [{"role": "user", "content": "test table"}]}],
    }
    spec = {
        "base_fingerprint": original,
        "rank": 1,
        "max_sequence_tokens": 64,
        "generation": {"max_new_tokens": 2, "max_context_tokens": 64},
        "training": dict(
            steps=1,
            lr=0.001,
            seed=43,
            beta=0.1,
            loss_scale=32.0,
            minimum_loss_scale=1.0,
            preservation_weight=1.0,
            gradient_clip=1.0,
        ),
    }
    execute(model, tokenizer, data, spec, tmp_path / "run", lambda: None)
    assert fingerprint(model) == original
    assert json.loads((tmp_path / "run/complete.json").read_text()) == {
        "conditions": 5,
        "base_restored": True,
    }
    predictions = list((tmp_path / "run").rglob("confirmation.json"))
    assert len(predictions) == 5
    for p in predictions:
        row = json.loads(p.read_text())[0]
        assert row["id"] == "test" and 1 <= len(row["tokens"]) <= 2
    assert len(list((tmp_path / "run").rglob("adapter_model.safetensors"))) == 4
