"""Independently verify decoding, frozen controls, alias exposure and finite probabilities."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.paired_endpoint.run import verify_parent
from experiments.paired_mlp.data import messages, whole_hit
from experiments.paired_mlp.run import BANKS
from neuronpatch.supervision import chat_response

from .run import verify_probes


def probe_metrics(path, rows, recorded=None):
    z = load_file(str(path))["logits"].double()
    assert z.ndim == 2 and torch.isfinite(z).all()
    offset, result = 0, []
    if recorded is not None:
        assert [r["id"] for r in rows] == [r["id"] for r in recorded]
    for index, row in enumerate(rows):
        unique = list(dict.fromkeys(tuple(p[:i]) for p in row["paths"] for i in range(len(p))))
        assert row["prefixes"] == [list(p) for p in unique]
        values = z[offset : offset + len(unique)]
        logp = torch.nn.functional.log_softmax(values, -1)
        lookup = {p: i for i, p in enumerate(unique)}
        logs = [
            sum(logp[lookup[p[:i]], t] for i, t in enumerate(p))
            for p in dict.fromkeys(tuple(p) for p in row["paths"])
        ]
        mass = float(torch.stack(logs).logsumexp(0).exp())
        assert 0 <= mass <= 1 + 1e-10
        if recorded is not None:
            assert abs(mass - recorded[index]["accepted_mass"]) < 1e-10
        result.append({"id": row["id"], "accepted_mass": mass})
        offset += len(unique)
    assert offset == len(z)
    return result, len(z)


def suffix_probability(path, rows, tracked):
    z = load_file(str(path))["logits"]
    offset = 0
    for row in rows:
        if row["id"] == tracked["id"]:
            i = row["prefixes"].index(tracked["prefix"])
            return float(z[offset + i].double().softmax(-1)[tracked["token"]])
        offset += len(row["prefixes"])
    raise ValueError("tracked suffix row absent")


def verify(parent, probes, results, protocol, tokenizer):
    verify_parent(parent, protocol)
    verify_probes(probes, protocol)
    assert json.loads((results / "protocol.json").read_text()) == protocol
    data = json.loads((parent / "inputs.json").read_text())
    original_fits = json.loads((parent / "fits.json").read_text())
    records = json.loads((results / "records.json").read_text())
    assert sum(r["arm"] == "canonical_control" for r in records) == protocol["canonical_controls"]
    assert all(r["arm"] == "canonical_control" for r in records[: protocol["canonical_controls"]])
    assert [(r["case"], r["method"], r["arm"]) for r in records] == [
        (r["case"], r["method"], r["arm"]) for r in protocol["requests"]
    ]
    observations, vectors, answer_cache, base_mass = 0, 0, {}, {}
    cases = {c["case"]: c for c in data["cases"]}

    def decode(case, value):
        nonlocal observations
        assert set(value) == set(BANKS)
        for bank in BANKS:
            for row, answer in zip(case[bank], value[bank], strict=True):
                assert row["id"] == answer["id"]
                ids = answer["tokens"]
                assert (
                    ids
                    and len(ids) <= protocol["max_new_tokens"]
                    and tokenizer.eos_token_id not in ids[:-1]
                )
                assert answer["ended"] == (ids[-1] == tokenizer.eos_token_id)
                assert answer["truncated"] == (not answer["ended"])
                assert answer["ended"] or len(ids) == protocol["max_new_tokens"]
                assert answer["text"] == tokenizer.decode(ids, skip_special_tokens=True)
                assert answer["unexpected_special"] == any(
                    t in tokenizer.all_special_ids and t != tokenizer.eos_token_id for t in ids
                )
                observations += 1

    for c, case in cases.items():
        baseline = json.loads((results / f"case-{c}-base-answers.json").read_text())
        assert baseline == json.loads((parent / f"case-{c}-base.json").read_text())
        decode(case, baseline)
        base_mass[c], count = probe_metrics(
            results / f"case-{c}-base-contexts.safetensors",
            protocol["rows"][str(c)],
            json.loads((results / f"case-{c}-base-masses.json").read_text()),
        )
        assert torch.equal(
            load_file(str(results / f"case-{c}-base-contexts.safetensors"))["logits"],
            load_file(str(probes / f"case-{c}-base-contexts.safetensors"))["logits"],
        )
        vectors += count
        for i, row in enumerate(case["preserve"]):
            assert protocol["training_aliases"][str(c)][i] == row["answers"]
            assert protocol["eligibility"][str(c)][i] == whole_hit(
                baseline["preserve"][i], row["answers"]
            )
    summaries = []
    for record, request in zip(records, protocol["requests"], strict=True):
        c, method, arm = record["case"], record["method"], record["arm"]
        case, stem = cases[c], f"case-{c}-{method}-{arm}"
        outputs = json.loads((results / f"{stem}-answers.json").read_text())
        answer_cache[c, method, arm] = outputs
        decode(case, outputs)
        assert (
            record["reload_parameters_exact"]
            and record["reload_teacher_exact"]
            and record["base_restored"]
        )
        assert record["trainable_scalars"] == request["trainable_scalars"]
        assert len(record["history"]) == protocol["steps"]
        assert [r["step"] for r in record["gradient_diagnostics"]] == list(
            range(1, protocol["steps"] + 1)
        )
        for row in record["gradient_diagnostics"]:
            assert row["task_gradient_norm"] >= 0 and row["auxiliary_contribution_norm"] >= 0
            cosine = row["task_auxiliary_cosine"]
            assert cosine is None or -1 - 1e-10 <= cosine <= 1 + 1e-10
        generator = torch.Generator().manual_seed(protocol["seed"])
        order = [
            torch.randperm(len(bank), generator=generator).tolist()
            for bank in (case["train"], case["preserve"])
        ]
        for step, history in enumerate(record["history"]):
            assert history["step"] == step + 1 and history["example_indices"] == [
                o[step % len(o)] for o in order
            ]
        accepted_calls = (
            sum(
                protocol["eligibility"][str(c)][order[1][step % len(order[1])]]
                for step in range(protocol["steps"])
            )
            if arm == "accepted_set"
            else 0
        )
        fallback_calls = protocol["steps"] - accepted_calls if arm == "accepted_set" else 0
        assert record["auxiliary_calls"] == {
            "accepted_set": accepted_calls,
            "canonical_fallback": fallback_calls,
        }
        labels, lengths = [], []
        for bank in ("train", "preserve", "target_val"):
            for row in case[bank]:
                s = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                )
                path = s.input_ids[0, s.prompt_length :].tolist()
                labels.extend(path)
                lengths.append(len(path))
        tensors = {}
        for state in ("live", "merged"):
            value = load_file(str(results / f"{stem}-{state}-teacher.safetensors"))
            assert value["labels"].tolist() == labels and value["lengths"].tolist() == lengths
            assert torch.isfinite(value["logits"]).all()
            tensors[state] = value
        current_mass, count = probe_metrics(
            results / f"{stem}-contexts.safetensors", protocol["rows"][str(c)], record["masses"]
        )
        vectors += count
        filename = (
            "adapter_model.safetensors"
            if request["condition"]["family"] == "lora"
            else "patch.safetensors"
        )
        parameters = load_file(str(results / f"{stem}-adapter" / filename))
        assert sum(p.numel() for p in parameters.values()) == request["trainable_scalars"]
        if arm == "canonical_control":
            reference = next(
                r for r in original_fits if r["case"] == c and r["condition"]["id"] == method
            )
            assert record["history"] == reference["history"]
            old = load_file(str(parent / f"case-{c}-{method}-teacher.safetensors"))
            assert all(torch.equal(tensors["merged"][k], old[k]) for k in old)
            assert outputs == json.loads((parent / f"case-{c}-{method}.json").read_text())
            original = load_file(str(parent / f"case-{c}-{method}-adapter" / filename))
            assert parameters.keys() == original.keys() and all(
                torch.equal(parameters[k], original[k]) for k in parameters
            )
            assert torch.equal(
                load_file(str(results / f"{stem}-contexts.safetensors"))["logits"],
                load_file(str(probes / f"case-{c}-{method}-contexts.safetensors"))["logits"],
            )
        bank_metrics = {}
        base_outputs = json.loads((results / f"case-{c}-base-answers.json").read_text())
        for bank in BANKS:
            hits = [
                whole_hit(a, row["answers"])
                for row, a in zip(case[bank], outputs[bank], strict=True)
            ]
            old = [
                whole_hit(a, row["answers"])
                for row, a in zip(case[bank], base_outputs[bank], strict=True)
            ]
            bank_metrics[bank] = {
                "hits": sum(hits),
                "total": len(hits),
                "scored_lost": [
                    r["id"] for r, a, b in zip(case[bank], old, hits, strict=True) if a and not b
                ],
                "truncated": sum(a["truncated"] for a in outputs[bank]),
            }
        drops = [
            b["accepted_mass"] - a["accepted_mass"]
            for row, b, a in zip(protocol["rows"][str(c)], base_mass[c], current_mass, strict=True)
            if row["retention_eligible"]
        ]
        summaries.append(
            {
                "case": c,
                "method": method,
                "arm": arm,
                "banks": bank_metrics,
                "masses": current_mass,
                "maximum_eligible_mass_drop": max(drops),
                "fit_seconds": record["fit_seconds"],
                "peak_cuda_allocated_bytes": record["peak_cuda_allocated_bytes"],
            }
        )
    assert (
        observations == protocol["expected_generation_observations"]
        and vectors == protocol["expected_probe_vectors"]
    )
    completion = json.loads((results / "completion.json").read_text())
    assert completion == {
        "status": "complete",
        "new_fits": protocol["new_fits"],
        "observations": observations,
        "probe_vectors": vectors,
        "base_restored": True,
        "final_fingerprint": protocol["base_fingerprint"],
    }
    tracked = protocol["tracked_suffix"]
    causal_rows = []
    for method in sorted({r["method"] for r in records if r["arm"] == "accepted_set"}):
        c = tracked["case"]
        old = suffix_probability(
            probes / f"case-{c}-{method}-contexts.safetensors", protocol["rows"][str(c)], tracked
        )
        new = suffix_probability(
            results / f"case-{c}-{method}-accepted_set-contexts.safetensors",
            protocol["rows"][str(c)],
            tracked,
        )
        rows = [r for r in summaries if r["method"] == method and r["arm"] == "accepted_set"]
        transfers = all(
            r["banks"]["target_test"]["hits"] == r["banks"]["target_test"]["total"] for r in rows
        )
        probe_row = next(r for r in protocol["rows"][str(c)] if r["id"] == tracked["id"])
        answer = next(
            a
            for a in answer_cache[c, method, "accepted_set"][probe_row["source_bank"]]
            if a["id"] == tracked["id"]
        )
        oxygen = whole_hit(answer, probe_row["accepted_aliases"])
        causal_rows.append(
            {
                "method": method,
                "canonical_suffix_probability": old,
                "accepted_set_suffix_probability": new,
                "all_transfers_retained": transfers,
                "oxygen_answer_accepted": oxygen,
                "supports_primary_prediction": new <= 0.5 * old and transfers and oxygen,
            }
        )
    supported = [r for r in causal_rows if r["supports_primary_prediction"]]
    causal_pass = len(supported) >= 2 and any(r["method"] == "paired_mean-43-0" for r in supported)
    proof = {
        "complete": True,
        "fits": len(records),
        "decoded": observations,
        "probe_vectors": vectors,
        "canonical_controls_exact": protocol["canonical_controls"],
        "file_sha256": {
            str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in results.rglob("*")
            if p.is_file()
        },
        "scope": (
            "Saved final probabilities,decoding,weights,source references and canonical histories; "
            "intermediate accepted-set optimizer trajectories and model forwards are not "
            "independently replayed."
        ),
    }
    return proof, {
        "rows": summaries,
        "causal_comparison": causal_rows,
        "primary_causal_prediction_pass": causal_pass,
        "full_output_review_complete": False,
        "selection_advantage_established": False,
        "limits": protocol["limits"],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "probes", "results", "protocol", "integrity", "analysis"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"], revision=protocol["revision"], local_files_only=True
    )
    proof, analysis = verify(args.parent, args.probes, args.results, protocol, tokenizer)
    for path, value in ((args.integrity, proof), (args.analysis, analysis)):
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": proof["complete"],
                "fits": proof["fits"],
                "primary_causal_prediction_pass": analysis["primary_causal_prediction_pass"],
            }
        )
    )


if __name__ == "__main__":
    main()
