"""Run the exact ablation/export worker on small FP32/FP16 Qwen models."""

import hashlib
import json

import pytest
import torch
from test_paired_worker import TinyTokenizer, tiny_data

from experiments.alias_supervision.prepare import prepare
from experiments.alias_supervision.run import execute
from experiments.alias_supervision.verify import verify
from experiments.outcome_guard.prepare import token_spec
from experiments.outcome_guard.run import execute as replay
from experiments.outcome_guard.verify import verify as verify_replay
from experiments.paired_endpoint.prepare import prepare as endpoint_protocol
from experiments.paired_mlp.prepare import protocol as paired_protocol
from experiments.paired_mlp.run import execute as train_parent
from experiments.paired_mlp.verify import verify as verify_parent
from neuronpatch.patch import fingerprint


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_actual_eight_fit_ablation_recovers_controls_trains_aliases_and_exports(tmp_path, dtype):
    tf = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = tf.Qwen3Config(
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
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(54)
    model = tf.Qwen3ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    tokenizer = TinyTokenizer()
    red = tokenizer.encode("red")[0]
    tokenizer.encode("red.")
    # A deterministic useful base: other words favor red, and red favors EOS.
    # Transformer blocks keep their nonzero random weights, permitting adapter gradients.
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()
        model.model.embed_tokens.weight[:, 0] = 3
        model.model.embed_tokens.weight[red].zero_()
        model.model.embed_tokens.weight[red, 1] = 3
        model.lm_head.weight.zero_()
        model.lm_head.weight[red, 0] = 3
        model.lm_head.weight[2, 1] = 3
    original = fingerprint(model)
    data = tiny_data()
    for case in data["cases"]:
        case["preserve"][0]["answers"] = ["red", "red red"]
    spec = paired_protocol(hashlib.sha256((json.dumps(data, indent=2) + "\n").encode()).hexdigest())
    spec.update(
        architecture={"layers": 2, "hidden": 16, "intermediate": 32},
        minimum_base_correct_utility_test=0,
        steps=1,
        max_new_tokens=2,
    )
    parent, probes, output = [tmp_path / n for n in ("parent", "probes", "ablation")]
    train_parent(model, tokenizer, data, spec, parent, lambda: None)
    proof = verify_parent(parent, spec, tokenizer)
    parent_proof = tmp_path / "parent-proof.json"
    parent_proof.write_text(json.dumps(proof))
    probe_spec = endpoint_protocol(parent, parent_proof)
    probe_spec["endpoints"] = [e for e in probe_spec["endpoints"] if e["condition"]["lr"] == 0.0003]
    probe_rows = {}
    for case in data["cases"]:
        c = case["case"]
        values = []
        for bank in ("preserve", "utility_test"):
            value = token_spec(tokenizer, case[bank][0], 128)
            value.update(source_bank=bank, retention_eligible=True)
            values.append(value)
        probe_rows[str(c)] = values
    probe_spec.update(
        rows=probe_rows,
        expected_endpoints=6,
        expected_generations=24,
        maximum_absolute_mass_drop=0.02,
        positive_methods=["lora-43-0", "paired_random-11-0"],
        negative={
            "case": 1,
            "method": "paired_mean-43-0",
            "ids": [r["id"] for r in probe_rows["1"]],
        },
    )
    replay(model, tokenizer, parent, probe_spec, probes, lambda: None)
    qp, _ = verify_replay(parent, probes, probe_spec, tokenizer)
    probe_proof = tmp_path / "probe-proof.json"
    probe_proof.write_text(json.dumps(qp))
    ablation = prepare(parent, parent_proof, probes, probe_proof)
    ablation["tracked_suffix"] = {
        "case": 1,
        "id": probe_rows["1"][-1]["id"],
        "prefix": [red],
        "token": red,
    }
    records = execute(model, tokenizer, parent, probes, ablation, output, lambda: None)
    assert len(records) == 8 and fingerprint(model) == original
    assert all(r["canonical_control_exact"] for r in records[:2])
    assert all(r["auxiliary_calls"]["accepted_set"] == 1 for r in records[2:])
    assert all(r["reload_parameters_exact"] and r["reload_teacher_exact"] for r in records)
    checked, analysis = verify(parent, probes, output, ablation, tokenizer)
    assert checked["complete"] and checked["fits"] == 8 and checked["canonical_controls_exact"] == 2
    assert checked["decoded"] == ablation["expected_generation_observations"]
    assert not analysis["selection_advantage_established"]
    from experiments.alias_supervision.gradient_audit import audit
    from experiments.alias_supervision.review import packet

    integrity = tmp_path / "ablation-proof.json"
    integrity.write_text(json.dumps(checked))
    gradients = audit(output, integrity)
    assert len(gradients["rows"]) == 8
    review = packet(parent, output)
    assert review["observations"] == checked["decoded"]
    assert sum(len(r["occurrences"]) for r in review["items"]) == checked["decoded"]
    assert not review["full_output_review_complete"]
    # A forged marginal-probability result must fail independent verification.
    path = output / "records.json"
    changed = json.loads(path.read_text())
    changed[-1]["masses"][0]["accepted_mass"] += 0.01
    path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        verify(parent, probes, output, ablation, tokenizer)
