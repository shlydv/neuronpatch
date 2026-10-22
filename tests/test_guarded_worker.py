import hashlib
import json

import pytest
import torch
from test_paired_worker import TinyTokenizer, tiny_data

from experiments.guarded_training.run import execute
from experiments.guarded_training.verify import verify
from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import layout
from experiments.paired_mlp.run import generate
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_all_guarded_representations_fit_reload_merge_and_restore_the_actual_tiny_qwen_worker(
    tmp_path, dtype
):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(54)
    model = transformers.Qwen3ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    original = fingerprint(model)
    parent = tmp_path / "parent"
    parent.mkdir()
    data = tiny_data()
    tokenizer = TinyTokenizer()
    (parent / "inputs.json").write_text(json.dumps(data))
    (parent / "identity.json").write_text(json.dumps({"base_fingerprint": original}))
    banks = ["target_test", "scope_test", "utility_test"]
    samples, tokens = {}, {}
    for case in data["cases"]:
        for bank in ("train", "preserve", *banks):
            for row in case[bank]:
                sample = chat_response(tokenizer, messages(row), row["answers"][0], max_length=32)
                samples[row["id"]] = sample
                tokens[row["id"]] = {
                    "ids": sample.input_ids[0].tolist(),
                    "prompt_length": sample.prompt_length,
                }
    (parent / "tokenization.json").write_text(json.dumps(tokens))
    for case in data["cases"]:
        baseline = {
            b: [generate(model, tokenizer, samples[r["id"]], r, 2) for r in case[b]] for b in banks
        }
        (parent / f"case-{case['case']}-base.json").write_text(json.dumps(baseline))
    groups, counts, _ = layout(model, 1)
    methods = ["gradient_norm", "gain_risk", "random", "lora"]
    support = {
        str(c["case"]): {
            method: [
                {
                    "incoming": g["incoming"],
                    "outgoing": g["outgoing"],
                    "indices": list(range(3 * i, 3 * i + n)),
                }
                for g, n in zip(groups, counts, strict=True)
            ]
            for i, method in enumerate(methods[:-1])
        }
        for c in data["cases"]
    }
    protocol = {
        "id": "tiny-guarded-worker",
        "parent_files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()
        },
        "base_fingerprint": original,
        "architecture": {"layers": 2, "hidden": 16, "intermediate": 32},
        "rank": 1,
        "seed": 43,
        "methods": methods,
        "supports": support,
        "steps": 1,
        "learning_rate": 0.01,
        "budget": {"mean_kl": 100.0, "worst_kl": 100.0, "minimum_improvement": 1e-7},
        "preservation_weight": 1.0,
        "loss_scale": 128.0,
        "gradient_clip": 1.0,
        "confidence": 0.9,
        "backtrack_scales": [1.0, 0.5, 0.25, 0.125],
        "max_sequence_tokens": 32,
        "max_new_tokens": 2,
        "generation_banks": banks,
        "heldout_teacher_banks": ["scope_test", "utility_test"],
        "expected_fits": 8,
        "expected_observations": 78,
        "expected_base_generation_controls": 8,
    }
    protocol = json.loads(json.dumps(protocol))
    records = execute(model, tokenizer, parent, protocol, tmp_path / "result", lambda: None)
    assert len(records) == 8
    assert {r["trainable_scalars"] for r in records} == {288}
    assert all(r["fit"]["accepted_updates"] == 1 for r in records)
    assert all(
        r["reload_teacher_exact"] and r["reload_generations_exact"] and r["base_restored"]
        for r in records
    )
    assert all(r["fit"]["final"]["task_nll"] < r["fit"]["initial"]["task_nll"] for r in records)
    assert fingerprint(model) == original
    proof = verify(parent, tmp_path / "result", protocol, tokenizer)
    assert (
        proof["fits"] == 8 and proof["decoded"] == 78 and proof["teacher_records_recomputed"] == 18
    )
    # A false accepted update must be rejected independently of its summary flag.
    path = tmp_path / "result/fits.json"
    corrupted = json.loads(path.read_text())
    corrupted[0]["fit"]["history"][0]["attempts"][0]["measurement"]["task_nll"] = 100.0
    path.write_text(json.dumps(corrupted))
    with pytest.raises(AssertionError):
        verify(parent, tmp_path / "result", protocol, tokenizer)
