"""Recompute every replay loss/divergence and decode both endpoint states."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from .run import verify_parent


def verify(parent, results, protocol, tokenizer):
    verify_parent(parent, protocol)
    assert json.loads((results / "protocol.json").read_text()) == protocol
    data = json.loads((parent / "inputs.json").read_text())
    cases = {c["case"]: c for c in data["cases"]}
    records = json.loads((results / "endpoints.json").read_text())
    expected = {(r["case"], r["condition"]["id"]) for r in protocol["endpoints"]}
    assert len(records) == len(expected) == 8
    assert {(r["case"], r["condition"]["id"]) for r in records} == expected
    observations = 0
    for record in records:
        c, name = record["case"], record["condition"]["id"]
        prefix = f"case-{c}-{name}"
        case = cases[c]
        assert record["base_restored"] and record["parent_merged_teacher_exact"]
        assert record["parent_merged_generation_exact"]
        saved = load_file(str(parent / f"{prefix}-teacher.safetensors"))
        teachers, states = {}, {}
        for state in ("live", "merged"):
            t = load_file(str(results / f"{prefix}-{state}-teacher.safetensors"))
            assert set(t) == set(saved)
            assert torch.equal(t["labels"], saved["labels"])
            assert torch.equal(t["lengths"], saved["lengths"])
            assert t["logits"].shape == saved["logits"].shape
            assert torch.isfinite(t["logits"]).all()
            if state == "merged":
                assert all(torch.equal(t[k], saved[k]) for k in saved)
            loss = F.cross_entropy(t["logits"].double(), t["labels"], reduction="none")
            parts = loss.split(t["lengths"].tolist())
            n, p = len(case["train"]), len(case["preserve"])
            measured = {
                "task_mean_nll": sum(float(x.mean()) for x in parts[:n]) / n,
                "task_worst_token_nll": max(float(x.max()) for x in parts[:n]),
                "preserve_mean_nll": sum(float(x.mean()) for x in parts[n : n + p]) / p,
                "target_val_mean_nll": sum(float(x.mean()) for x in parts[n + p :])
                / len(parts[n + p :]),
            }
            assert all(
                math.isclose(v, record[f"{state}_metrics"][k], abs_tol=1e-10, rel_tol=1e-8)
                for k, v in measured.items()
            )
            outputs = json.loads((results / f"{prefix}-{state}.json").read_text())
            assert set(outputs) == set(protocol["generation_banks"])
            for bank, answers in outputs.items():
                assert len(answers) == len(case[bank])
                for row, answer in zip(case[bank], answers, strict=True):
                    assert answer["id"] == row["id"]
                    tokens = answer["tokens"]
                    assert 0 < len(tokens) <= protocol["max_new_tokens"]
                    assert tokenizer.eos_token_id not in tokens[:-1]
                    assert answer["text"] == tokenizer.decode(tokens, skip_special_tokens=True)
                    ended = tokens[-1] == tokenizer.eos_token_id
                    assert answer["ended"] == ended and answer["truncated"] == (not ended)
                    assert ended or len(tokens) == protocol["max_new_tokens"]
                    assert answer["unexpected_special"] == any(
                        x in tokenizer.all_special_ids and x != tokenizer.eos_token_id
                        for x in tokens
                    )
                    observations += 1
            teachers[state], states[state] = t, outputs
        parent_outputs = json.loads((parent / f"{prefix}.json").read_text())
        assert all(states["merged"][b] == parent_outputs[b] for b in protocol["generation_banks"])
        a, b = (teachers[s]["logits"].double() for s in ("live", "merged"))
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        metrics = record["teacher_comparison"]
        assert metrics["logits_exact"] == torch.equal(
            teachers["live"]["logits"], teachers["merged"]["logits"]
        )
        assert metrics["teacher_positions"] == len(a)
        assert metrics["changed_teacher_argmax_positions"] == int(
            (a.argmax(-1) != b.argmax(-1)).sum()
        )
        for key, value in (
            ("mean_kl_live_to_merged", F.kl_div(lb, la, reduction="batchmean", log_target=True)),
            ("mean_kl_merged_to_live", F.kl_div(la, lb, reduction="batchmean", log_target=True)),
            ("max_abs_logit_difference", (a - b).abs().max()),
        ):
            # KL can be below1e-8; that absolute tolerance would accept a false zero.
            assert math.isclose(metrics[key], float(value), rel_tol=1e-8, abs_tol=1e-12)
        assert record["changed_generations"] == {
            bank: [
                a["id"]
                for a, b in zip(states["live"][bank], states["merged"][bank], strict=True)
                if a != b
            ]
            for bank in protocol["generation_banks"]
        }
    completed = json.loads((results / "completion.json").read_text())
    assert completed == {
        "status": "complete",
        "new_fits": 0,
        "endpoints": len(records),
        "observations": observations,
        "base_restored": True,
        "final_fingerprint": protocol["base_fingerprint"],
    }
    assert observations == protocol["expected_generations"]
    return {
        "complete": True,
        "new_fits": 0,
        "endpoints": len(records),
        "decoded": observations,
        "parent_merged_teacher_exact": True,
        "parent_merged_generations_exact": True,
        "file_sha256": {
            str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in results.rglob("*")
            if p.is_file()
        },
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    proof = verify(args.parent, args.results, protocol, tokenizer)
    with args.output.open("x") as out:
        out.write(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "file_sha256"}))


if __name__ == "__main__":
    main()
