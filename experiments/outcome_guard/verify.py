"""Independently recompute finite-event probabilities and the frozen qualification."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.paired_endpoint.run import verify_parent
from experiments.paired_mlp.data import messages, whole_hit
from neuronpatch.supervision import chat_response


def assess(records, answers, cases, protocol):
    by_state = {(r["case"], r["method"]): r for r in records}
    rows, positives = [], []
    for case in cases:
        c = case["case"]
        base = {r["id"]: r["accepted_mass"] for r in by_state[c, "base"]["masses"]}
        for endpoint in [e for e in protocol["endpoints"] if e["case"] == c]:
            method = endpoint["condition"]["id"]
            state = by_state[c, method]
            mass = {r["id"]: r["accepted_mass"] for r in state["masses"]}
            outputs = {r["id"]: r for r in answers[c, method]["probes"]}
            probes = []
            for row in protocol["rows"][str(c)]:
                drop = base[row["id"]] - mass[row["id"]]
                probes.append(
                    {
                        "id": row["id"],
                        "retention_eligible": row["retention_eligible"],
                        "base_mass": base[row["id"]],
                        "edited_mass": mass[row["id"]],
                        "drop": drop,
                        "probability_flag": drop > protocol["maximum_absolute_mass_drop"],
                        "answer_accepted": whole_hit(outputs[row["id"]], row["accepted_aliases"]),
                    }
                )
            eligible = [r for r in probes if r["retention_eligible"]]
            probability_accept = bool(eligible) and not any(r["probability_flag"] for r in eligible)
            answer_accept = bool(eligible) and all(r["answer_accepted"] for r in eligible)
            target = answers[c, method]["target_test"]
            transfer = all(
                whole_hit(a, r["answers"]) for r, a in zip(case["target_test"], target, strict=True)
            )
            value = {
                "case": c,
                "method": method,
                "probes": probes,
                "probability_accept": probability_accept,
                "answer_accept": answer_accept,
                "complete_transfer": transfer,
            }
            rows.append(value)
            if method in protocol["positive_methods"]:
                positives.append(probability_accept and answer_accept and transfer)
    negative = protocol["negative"]
    candidate = next(
        r for r in rows if (r["case"], r["method"]) == (negative["case"], negative["method"])
    )
    checks = [next(r for r in candidate["probes"] if r["id"] == key) for key in negative["ids"]]
    assert len(positives) == 2 * len(cases)
    flags = [r["retention_eligible"] and r["probability_flag"] for r in checks]
    return {
        "rows": rows,
        "all_required_positive_controls_pass": all(positives),
        "all_required_negative_rows_probability_flagged": all(flags),
        "all_required_negative_rows_answer_rejected": all(not r["answer_accepted"] for r in checks),
        "qualification_pass": all(positives)
        and all(flags)
        and all(not r["answer_accepted"] for r in checks),
        "limits": protocol["limits"],
    }


def verify(parent, results, protocol, tokenizer):
    verify_parent(parent, protocol)
    assert json.loads((results / "protocol.json").read_text()) == protocol
    cases = json.loads((parent / "inputs.json").read_text())["cases"]
    records = json.loads((results / "records.json").read_text())
    expected = {(c["case"], "base") for c in cases} | {
        (e["case"], e["condition"]["id"]) for e in protocol["endpoints"]
    }
    assert len(records) == len(expected) and {(r["case"], r["method"]) for r in records} == expected
    answers, observations, prefixes = {}, 0, 0
    for case in cases:
        c = case["case"]
        labels, lengths = [], []
        for bank in ("train", "preserve", "target_val"):
            for row in case[bank]:
                s = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                )
                p = s.input_ids[0, s.prompt_length :].tolist()
                labels.extend(p)
                lengths.append(len(p))
        for state in [r for r in records if r["case"] == c]:
            method = state["method"]
            stem = f"case-{c}-{method}"
            old = json.loads((parent / f"{stem}.json").read_text())
            output = json.loads((results / f"{stem}-answers.json").read_text())
            answers[c, method] = output
            assert set(output) == {"probes", "target_test"}
            for bank, rows in (
                ("probes", protocol["rows"][str(c)]),
                ("target_test", case["target_test"]),
            ):
                for row, answer in zip(rows, output[bank], strict=True):
                    source = row["source_bank"] if bank == "probes" else bank
                    assert answer == next(a for a in old[source] if a["id"] == row["id"])
                    ids = answer["tokens"]
                    assert ids and len(ids) <= protocol["max_new_tokens"]
                    assert tokenizer.eos_token_id not in ids[:-1]
                    assert answer["text"] == tokenizer.decode(ids, skip_special_tokens=True)
                    assert answer["ended"] == (ids[-1] == tokenizer.eos_token_id)
                    assert answer["truncated"] == (not answer["ended"])
                    assert answer["ended"] or len(ids) == protocol["max_new_tokens"]
                    assert answer["unexpected_special"] == any(
                        t in tokenizer.all_special_ids and t != tokenizer.eos_token_id for t in ids
                    )
                    observations += 1
            teacher = load_file(str(results / f"{stem}-teacher.safetensors"))
            assert teacher["labels"].tolist() == labels and teacher["lengths"].tolist() == lengths
            if method != "base":
                previous = load_file(str(parent / f"{stem}-teacher.safetensors"))
                assert all(torch.equal(teacher[k], previous[k]) for k in previous)
                assert state["base_restored"] and state["parent_teacher_exact"]
            tensors = load_file(str(results / f"{stem}-contexts.safetensors"))
            assert set(tensors) == {"logits"}
            logits = tensors["logits"]
            assert logits.ndim == 2 and torch.isfinite(logits).all()
            offset = 0
            assert [r["id"] for r in state["masses"]] == [r["id"] for r in protocol["rows"][str(c)]]
            for row, measured in zip(protocol["rows"][str(c)], state["masses"], strict=True):
                unique = list(
                    dict.fromkeys(tuple(p[:i]) for p in row["paths"] for i in range(len(p)))
                )
                assert row["prefixes"] == [list(p) for p in unique]
                for alias, path in zip(row["accepted_aliases"], row["paths"], strict=True):
                    s = chat_response(
                        tokenizer, messages(row), alias, max_length=protocol["max_sequence_tokens"]
                    )
                    assert s.input_ids[0, : s.prompt_length].tolist() == row["prompt_tokens"]
                    assert s.input_ids[0, s.prompt_length :].tolist() == path
                    assert (
                        path[-1] == tokenizer.eos_token_id
                        and tokenizer.eos_token_id not in path[:-1]
                    )
                z = logits[offset : offset + len(unique)].double()
                offset += len(unique)
                lookup = {p: i for i, p in enumerate(unique)}
                logp = torch.nn.functional.log_softmax(z, dim=-1)
                # Independent explicit sequence products; no response-set helper.
                path_logs = [
                    sum(logp[lookup[p[:i]], token] for i, token in enumerate(p))
                    for p in dict.fromkeys(tuple(p) for p in row["paths"])
                ]
                mass = float(torch.logsumexp(torch.stack(path_logs), dim=0).exp())
                assert 0 <= mass <= 1 + 1e-10
                assert abs(mass - measured["accepted_mass"]) < 1e-10
                baseline = json.loads((parent / f"case-{c}-base.json").read_text())
                original = next(a for a in baseline[row["source_bank"]] if a["id"] == row["id"])
                assert row["retention_eligible"] == whole_hit(original, row["answers"])
                measured["accepted_mass"] = mass
            assert offset == len(logits) == state["prefix_forwards"]
            prefixes += offset
    assert observations == protocol["expected_generations"]
    completion = json.loads((results / "completion.json").read_text())
    assert completion == {
        "status": "complete",
        "new_fits": 0,
        "states": len(records),
        "observations": observations,
        "base_controls": 2 * protocol["expected_endpoints"],
        "base_restored": True,
        "final_fingerprint": protocol["base_fingerprint"],
    }
    return {
        "complete": True,
        "states": len(records),
        "generated_observations": observations,
        "independently_recomputed_prefix_vectors": prefixes,
        "file_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in results.iterdir()
            if p.is_file()
        },
        "scope": (
            "Saved-logit probability arithmetic, complete generation replay, canonical "
            "teacher replay and declared base restoration. Does not independently rerun "
            "every prefix forward or optimizer (none exists)."
        ),
    }, assess(records, answers, cases, protocol)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "protocol", "integrity", "analysis"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"], revision=protocol["revision"], local_files_only=True
    )
    proof, analysis = verify(args.parent, args.results, protocol, tokenizer)
    for path, value in ((args.integrity, proof), (args.analysis, analysis)):
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in analysis.items() if k not in ("rows", "limits")}, indent=2))


if __name__ == "__main__":
    main()
