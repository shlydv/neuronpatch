"""Execute the actual replay on tiny Qwen endpoints and reject corrupted results."""

import copy
import hashlib
import json

import pytest
import torch
from test_paired_worker import TinyTokenizer, tiny_data

from experiments.outcome_guard.prepare import token_spec
from experiments.outcome_guard.run import collect, execute
from experiments.outcome_guard.verify import assess, verify
from experiments.paired_endpoint.prepare import prepare as replay_protocol
from experiments.paired_mlp.data import whole_hit
from experiments.paired_mlp.prepare import protocol as training_protocol
from experiments.paired_mlp.run import execute as create_parent
from experiments.paired_mlp.verify import verify as verify_parent
from neuronpatch.patch import fingerprint


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_actual_no_fit_worker_covers_alias_branches_replays_and_restores(tmp_path, dtype):
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
    )
    config._attn_implementation = "eager"
    torch.manual_seed(54)
    model = tf.Qwen3ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    original = fingerprint(model)
    data = tiny_data()
    tokenizer = TinyTokenizer()
    tokenizer.encode("red red.")  # Freeze fixture vocabulary before golden generations.
    spec = training_protocol(
        hashlib.sha256((json.dumps(data, indent=2) + "\n").encode()).hexdigest()
    )
    spec.update(
        architecture={"layers": 2, "hidden": 16, "intermediate": 32},
        minimum_base_correct_utility_test=0,
        steps=1,
        max_new_tokens=2,
    )
    parent, results = tmp_path / "parent", tmp_path / "results"
    create_parent(model, tokenizer, data, spec, parent, lambda: None)
    proof = verify_parent(parent, spec, tokenizer)
    integrity = tmp_path / "integrity.json"
    integrity.write_text(json.dumps(proof))
    spec = replay_protocol(parent, integrity)
    spec["endpoints"] = [e for e in spec["endpoints"] if e["condition"]["lr"] == 0.0003]
    rows = {}
    for case in data["cases"]:
        baseline = json.loads((parent / f"case-{case['case']}-base.json").read_text())
        rows[str(case["case"])] = []
        for bank in ("preserve", "utility_test"):
            row = case[bank][0]
            value = token_spec(tokenizer, {**row, "answers": ["red", "red red"]}, 128)
            value["answers"] = row["answers"]
            value.update(
                source_bank=bank, retention_eligible=whole_hit(baseline[bank][0], row["answers"])
            )
            rows[str(case["case"])].append(value)
    spec.update(
        rows=rows,
        expected_endpoints=6,
        expected_generations=24,
        maximum_absolute_mass_drop=0.02,
        positive_methods=["lora-43-0", "paired_random-11-0"],
        negative={"case": 1, "method": "paired_mean-43-0", "ids": [r["id"] for r in rows["1"]]},
    )
    records = execute(model, tokenizer, parent, spec, results, lambda: None)
    assert len(records) == 8 and fingerprint(model) == original
    verified, analysis = verify(parent, results, spec, tokenizer)
    assert verified["generated_observations"] == 24
    assert verified["independently_recomputed_prefix_vectors"] > 24
    assert not analysis["qualification_pass"]  # Random tiny model is not a quality witness.
    # One branch agrees with a direct independently executed prefix forward.
    logits, _ = collect(model, rows["0"][:1], tokenizer.eos_token_id, lambda: None)
    row = rows["0"][0]
    prefix = row["prefixes"][-1]
    ids = torch.tensor([row["prompt_tokens"] + prefix])
    with torch.no_grad():
        expected = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits[0, -1]
    torch.testing.assert_close(logits[-1], expected, rtol=0, atol=0)
    corrupted = json.loads((results / "records.json").read_text())
    corrupted[0]["masses"][0]["accepted_mass"] += 0.01
    (results / "records.json").write_text(json.dumps(corrupted))
    with pytest.raises(AssertionError):
        verify(parent, results, spec, tokenizer)


def test_qualification_requires_both_harmful_rows_and_all_positive_controls():
    cases, records, outputs, endpoints, rows = [], [], {}, [], {}
    for c in (0, 1):
        probes = [
            {
                "id": f"{c}-p{i}",
                "answers": ["ok"],
                "accepted_aliases": ["ok"],
                "retention_eligible": True,
            }
            for i in (0, 1)
        ]
        rows[str(c)] = probes
        cases.append({"case": c, "target_test": [{"answers": ["new"]}]})
        for method in ("base", "lora", "random", "selected"):
            records.append(
                {
                    "case": c,
                    "method": method,
                    "masses": [{"id": p["id"], "accepted_mass": 0.99} for p in probes],
                }
            )
            outputs[c, method] = {
                "probes": [
                    {"id": p["id"], "text": "ok", "ended": True, "unexpected_special": False}
                    for p in probes
                ],
                "target_test": [{"text": "new", "ended": True, "unexpected_special": False}],
            }
            if method != "base":
                endpoints.append({"case": c, "condition": {"id": method}})
    spec = {
        "endpoints": endpoints,
        "rows": rows,
        "positive_methods": ["lora", "random"],
        "maximum_absolute_mass_drop": 0.02,
        "limits": "fixture",
        "negative": {"case": 1, "method": "selected", "ids": [p["id"] for p in rows["1"]]},
    }
    negative = next(r for r in records if (r["case"], r["method"]) == (1, "selected"))
    for i in (0, 1):
        negative["masses"][i]["accepted_mass"] = 0.1
        outputs[1, "selected"]["probes"][i]["text"] = "wrong"
    assert assess(records, outputs, cases, spec)["qualification_pass"]
    missing = copy.deepcopy(records)
    next(r for r in missing if (r["case"], r["method"]) == (1, "selected"))["masses"][1][
        "accepted_mass"
    ] = 0.99
    assert not assess(missing, outputs, cases, spec)["qualification_pass"]
    damaged = copy.deepcopy(records)
    next(r for r in damaged if (r["case"], r["method"]) == (0, "lora"))["masses"][0][
        "accepted_mass"
    ] = 0.9
    assert not assess(damaged, outputs, cases, spec)["qualification_pass"]
