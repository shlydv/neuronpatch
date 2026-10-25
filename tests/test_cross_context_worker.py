import json

import torch
from test_paired_worker import TinyTokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.instruction_boundary.cross_context_run import execute
from neuronpatch.patch import fingerprint


def test_half_precision_worker_exports_both_objectives_and_restores_base(tmp_path):
    torch.manual_seed(31)
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
    model = Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    original = fingerprint(model)
    tok = TinyTokenizer()
    messages = [{"role": "user", "content": "train question"}]
    prompt = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    tokens = tok.encode("seven") + [2]
    task = {"document_id": "train", "messages": messages}
    teacher = {
        "id": "train:clean:none",
        "prompt_ids": prompt,
        "tokens": tokens,
        "text": "seven",
        "ended": True,
    }
    pairs = [
        {
            "id": f"pair-{i}-{trusted}",
            "document_id": "train",
            "trusted": trusted,
            "messages": [{"role": "user", "content": f"train question variant {i} {trusted}"}],
            "chosen": "marker" if trusted else "seven",
            "rejected": "seven" if trusted else "marker",
        }
        for i in range(3)
        for trusted in (False, True)
    ]
    data = {
        "train": {"tasks": [task], "diverse_pairs": pairs},
        "calibration": {"tasks": [{"document_id": "calibration"}]},
        "confirmation": {
            "tasks": [{"document_id": "confirmation"}],
            "cases": [{"id": "test", "messages": messages}],
        },
        "teachers": {
            "model": "tiny",
            "revision": "fixture",
            "recorded_base_fingerprint": original,
            "teachers": {"train": [{"task": task, "teacher": teacher}]},
        },
    }
    spec = {
        "model": "tiny",
        "revision": "fixture",
        "base_fingerprint": original,
        "rank": 1,
        "training_documents": 1,
        "diagnostic_case_ids": ["test"],
        "max_sequence_tokens": 64,
        "max_retention_cache_bytes": 10000,
        "generation": {"max_new_tokens": 2, "max_context_tokens": 64},
        "training": dict(
            steps=6,
            lr=0.001,
            seed=43,
            beta=0.1,
            loss_scale=32.0,
            minimum_loss_scale=1.0,
            preservation_weight=1.0,
            gradient_clip=1.0,
        ),
    }
    output = tmp_path / "run"
    execute(model, tok, data, spec, output, lambda: None)
    assert fingerprint(model) == original
    complete = json.loads((output / "comparison/complete.json").read_text())
    assert set(complete["arms"]) == {"diverse-sft-kl", "diverse-cross_context-kl"}
    for name, arm in complete["arms"].items():
        assert arm["successful_steps"] == 6
        assert arm["export_score_exact"] and arm["base_restored"]
        assert arm["forward_calls"] == 12
        history = json.loads((output / "comparison" / name / "history.json").read_text())
        if name == "diverse-cross_context-kl":
            assert sum(r["task_target"] == "clean_distribution" for r in history) == 3
            assert sum(r["task_target"] == "hard_response" for r in history) == 3
