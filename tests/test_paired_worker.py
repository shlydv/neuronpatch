import hashlib
import json

import pytest
import torch

from experiments.paired_mlp.prepare import protocol
from experiments.paired_mlp.review import packet
from experiments.paired_mlp.run import BANKS, execute
from experiments.paired_mlp.verify import verify
from neuronpatch.patch import fingerprint


class TinyTokenizer:
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def __init__(self):
        self.words = {}

    def encode(self, text):
        result = []
        for word in text.split():
            if word not in self.words:
                self.words[word] = len(self.words) + 3
            result.append(self.words[word])
        return result

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert not return_dict
        rows = messages if add_generation_prompt else messages[:-1]
        ids = [1, *self.encode(rows[-1]["content"])]
        return ids if add_generation_prompt else [*ids, *self.encode(messages[-1]["content"]), 2]

    def decode(self, ids, *, skip_special_tokens):
        mapping = {v: k for k, v in self.words.items()}
        return " ".join(
            mapping.get(i, f"v{i}")
            for i in ids
            if not skip_special_tokens or i not in self.all_special_ids
        )


def tiny_data():
    cases = []
    for c, target in enumerate(("blue", "green")):
        case = {"case": c, "target": target}
        for bank in BANKS:
            count = 3 if bank == "train" else 1
            case[bank] = [
                {
                    "id": f"{c}-{bank}-{i}",
                    "prompt": f"value c{c} {bank} n{i}",
                    "answers": [target if bank.startswith("target") or bank == "train" else "red"],
                }
                for i in range(count)
            ]
        cases.append(case)
    return {"cases": cases, "scope": "CPU fixture, no language-quality claim"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_actual_instruction_worker_all_families_rates_seeds_roundtrip_and_restore(tmp_path, dtype):
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
    data = tiny_data()
    spec = protocol(hashlib.sha256((json.dumps(data, indent=2) + "\n").encode()).hexdigest())
    spec.update(
        id="cpu-fixture",
        architecture={"layers": 2, "hidden": 16, "intermediate": 32},
        minimum_base_correct_utility_test=0,
        steps=2,
        max_new_tokens=2,
    )
    tokenizer = TinyTokenizer()
    rows = execute(model, tokenizer, data, spec, tmp_path / "run", lambda: None)
    assert len(rows) == 24
    assert {r["trainable_scalars"] for r in rows} == {288}
    assert all(r["base_restored"] and r["reload_teacher_exact"] for r in rows)
    assert all(len(r["history"]) == 2 for r in rows)
    assert fingerprint(model) == original
    verified = verify(tmp_path / "run", spec, tokenizer)
    assert verified["fits"] == verified["teacher_records_recomputed"] == 24
    assert verified["decoded"] == 286
    review = packet(tmp_path / "run")
    assert review["observations"] == 286
    assert sum(len(item["occurrences"]) for item in review["items"]) == 286
    assert not review["full_output_review_complete"]
    # The no-fit replay must independently recover these saved endpoints in both dtypes.
    from experiments.paired_endpoint.prepare import prepare as prepare_replay
    from experiments.paired_endpoint.run import execute as replay
    from experiments.paired_endpoint.verify import verify as verify_replay

    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text(json.dumps(verified))
    replay_spec = prepare_replay(tmp_path / "run", integrity_path)
    endpoints = replay(
        model, tokenizer, tmp_path / "run", replay_spec, tmp_path / "replay", lambda: None
    )
    assert len(endpoints) == 8
    assert all(
        r["parent_merged_teacher_exact"] and r["parent_merged_generation_exact"] for r in endpoints
    )
    assert all(r["base_restored"] for r in endpoints)
    assert fingerprint(model) == original
    completed = json.loads((tmp_path / "replay/completion.json").read_text())
    assert completed["new_fits"] == 0 and completed["observations"] == 48
    replay_proof = verify_replay(tmp_path / "run", tmp_path / "replay", replay_spec, tokenizer)
    assert replay_proof["endpoints"] == 8 and replay_proof["decoded"] == 48
    from experiments.paired_endpoint.analysis import summarize as replay_summary
    from experiments.paired_endpoint.review import packet as replay_packet

    scored = replay_summary(tmp_path / "run", tmp_path / "replay")
    assert scored["new_fits"] == 0 and scored["observations"] == 48
    assert len(scored["cases"]) == 8 and len(scored["conditions"]) == 4
    reviewed = replay_packet(tmp_path / "run", tmp_path / "replay")
    assert reviewed["observations"] == 48 and not reviewed["full_output_review_complete"]
    assert sum(len(r["occurrences"]) for r in reviewed["items"]) == 48
    from experiments.paired_checkpoints.prepare import prepare as prepare_checkpoints
    from experiments.paired_checkpoints.run import execute as trace_checkpoints
    from experiments.paired_checkpoints.verify import verify as verify_checkpoints

    replay_integrity_path = tmp_path / "replay-integrity.json"
    replay_integrity_path.write_text(json.dumps(replay_proof))
    trace_spec = prepare_checkpoints(
        tmp_path / "run", integrity_path, tmp_path / "replay", replay_integrity_path
    )
    trace_spec.update(id="cpu-checkpoint-fixture", checkpoint_steps=[1, 2], gate_probability=1e-6)
    parents = {"original": tmp_path / "run", "replay": tmp_path / "replay"}
    traces = trace_checkpoints(
        model, tokenizer, parents, trace_spec, tmp_path / "trace", lambda: None
    )
    assert len(traces) == 7 and all(r["selected_step"] == 1 for r in traces)
    assert all(r["final_history_exact"] and r["final_parameters_exact"] for r in traces)
    proof = verify_checkpoints(parents, tmp_path / "trace", trace_spec, tokenizer)
    assert proof["fits"] == 7 and proof["decoded"] == 42 and proof["adam_steps_reconstructed"] == 1
    assert fingerprint(model) == original
    from experiments.paired_checkpoints.analysis import summarize as checkpoint_summary
    from experiments.paired_checkpoints.review import packet as checkpoint_packet

    summary = checkpoint_summary(tmp_path / "run", tmp_path / "trace")
    assert summary["fits"] == 7 and len(summary["conditions"]) == 3
    assert all(r["selected_steps"] == [1, 1] for r in summary["conditions"].values())
    review = checkpoint_packet(tmp_path / "run", tmp_path / "trace")
    assert review["observations"] == 42 and not review["full_output_review_complete"]
    # Failing the confidence gate must produce no invented early candidate.
    no_gate = {
        **trace_spec,
        "requests": trace_spec["requests"][:1],
        "new_fits": 1,
        "gate_probability": 1.0,
    }
    empty = trace_checkpoints(
        model, tokenizer, parents, no_gate, tmp_path / "no-gate", lambda: None
    )
    assert empty[0]["selected_step"] is None and set(empty[0]["states"]) == {"full"}
    empty_proof = verify_checkpoints(parents, tmp_path / "no-gate", no_gate, tokenizer)
    assert empty_proof["fits"] == 1 and empty_proof["decoded"] == 3
    assert not (tmp_path / "no-gate/case-0-paired_mean-43-0-early-adapter").exists()
    trace_records = tmp_path / "trace/fits.json"
    changed = json.loads(trace_records.read_text())
    changed[0]["selected_step"] = 2
    trace_records.write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        verify_checkpoints(parents, tmp_path / "trace", trace_spec, tokenizer)
    record_path = tmp_path / "replay/endpoints.json"
    altered = json.loads(record_path.read_text())
    altered[0]["live_metrics"]["task_mean_nll"] += 0.5
    record_path.write_text(json.dumps(altered))
    with pytest.raises(AssertionError):
        verify_replay(tmp_path / "run", tmp_path / "replay", replay_spec, tokenizer)
    path = tmp_path / "run/fits.json"
    fits = json.loads(path.read_text())
    fits[0]["metrics"]["task_mean_nll"] += 0.01
    path.write_text(json.dumps(fits))
    with pytest.raises(AssertionError):
        verify(tmp_path / "run", spec, tokenizer)
