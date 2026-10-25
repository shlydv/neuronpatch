import copy
import hashlib
import json

import pytest
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
    spec["arms"] = ["narrow-sft", "narrow-dpo", "diverse-sft", "diverse-dpo"]
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

    from experiments.instruction_boundary.verify_preference import verify

    receipt, outputs = verify(tmp_path / "run", data, spec, tokenizer)
    assert receipt["fits"] == 4 and receipt["generations"] == 5
    from experiments.instruction_boundary.loss_isolation_run import (
        compare_replay,
        find_prior,
    )
    from experiments.instruction_boundary.loss_isolation_run import (
        execute as isolate,
    )

    prior = tmp_path / "run/comparison/diverse-sft"
    diagnostic_spec = copy.deepcopy(spec)
    diagnostic_spec["prior_files"] = {
        name: hashlib.sha256((prior / name).read_bytes()).hexdigest()
        for name in (
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "confirmation.json",
        )
    }
    diagnostic_spec["training"]["steps"] = 2
    diagnostic_spec["diagnostic_steps"] = [1]
    diagnostic_spec["diagnostic_case_ids"] = ["test"]
    diagnostic_data = copy.deepcopy(data)
    diagnostic_data["confirmation"]["cases"][0]["kind"] = "clean"
    torch.manual_seed(31)
    fresh = Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    assert find_prior(tmp_path, diagnostic_spec["prior_files"]) == prior
    isolate(
        fresh,
        tokenizer,
        diagnostic_data,
        diagnostic_spec,
        tmp_path / "isolation",
        lambda: None,
        prior=prior,
    )
    assert fingerprint(fresh) == original
    completed = json.loads((tmp_path / "isolation/complete.json").read_text())
    assert completed["fits"] == 2 and completed["exact_prior_output_replay"]
    assert len(list((tmp_path / "isolation").rglob("clean-step-1.json"))) == 2
    assert len(list((tmp_path / "isolation").rglob("adapter_model.safetensors"))) == 2
    expected = json.loads((prior / "confirmation.json").read_text())
    altered = copy.deepcopy(expected)
    altered[0]["tokens"][0] = (altered[0]["tokens"][0] + 1) % 64
    with pytest.raises(ValueError, match="replay differs"):
        compare_replay(altered, expected, ["test"])

    path = tmp_path / "run/comparison/narrow-dpo/history.json"
    history = json.loads(path.read_text())
    history[0]["pair_id"] = "wrong-pair"
    path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="Wrong training pair"):
        verify(tmp_path / "run", data, spec, tokenizer)
