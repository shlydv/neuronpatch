"""Freeze the same original examples, supports and optimizer; vary auxiliary loss only."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.data import messages, whole_hit
from experiments.paired_mlp.run import BANKS
from neuronpatch.supervision import chat_response


def prepare(parent, parent_proof, probes, probe_proof):
    pp, qp = [json.loads(p.read_text()) for p in (parent_proof, probe_proof)]
    assert pp["complete"] and qp["complete"]
    probe_spec = json.loads((probes / "protocol.json").read_text())
    spec = copy.deepcopy(probe_spec)
    data = json.loads((parent / "inputs.json").read_text())
    original = json.loads((parent / "identity.json").read_text())["protocol"]
    fits = json.loads((parent / "fits.json").read_text())
    chosen = ("paired_mean-43-0", "paired_random-11-0", "lora-43-0")
    requests = []
    for method in ("paired_mean-43-0", "lora-43-0"):
        requests.append({"case": 1, "method": method, "arm": "canonical_control"})
    for case in data["cases"]:
        requests += [{"case": case["case"], "method": m, "arm": "accepted_set"} for m in chosen]
    eligibility, aliases = {}, {}
    for case in data["cases"]:
        c = case["case"]
        base = json.loads((parent / f"case-{c}-base.json").read_text())
        eligibility[str(c)] = [
            whole_hit(a, r["answers"])
            for r, a in zip(case["preserve"], base["preserve"], strict=True)
        ]
        aliases[str(c)] = [r["answers"] for r in case["preserve"]]
        for method in chosen:
            row = next(r for r in fits if r["case"] == c and r["condition"]["id"] == method)
            for request in [r for r in requests if r["case"] == c and r["method"] == method]:
                request.update(
                    condition=row["condition"],
                    supports=row["supports"],
                    trainable_scalars=row["trainable_scalars"],
                )
    for name, digest in probe_spec["parent_files_sha256"].items():
        assert pp["file_sha256"][name] == digest
        assert hashlib.sha256((parent / name).read_bytes()).hexdigest() == digest
    probe_hashes = qp["file_sha256"]
    assert all(
        hashlib.sha256((probes / n).read_bytes()).hexdigest() == h for n, h in probe_hashes.items()
    )
    spec.update(
        id="alias-supervision-ablation-v1-20261023",
        requests=requests,
        probe_kernel="shlydv/neuronpatch-outcome-guard-v1-20261023",
        probe_files_sha256=probe_hashes,
        eligibility=eligibility,
        training_aliases=aliases,
        seed=original["seed"],
        steps=original["steps"],
        preservation_weight=original["preservation_weight"],
        loss_scale=original["loss_scale"],
        gradient_clip=original["gradient_clip"],
        new_fits=len(requests),
        canonical_controls=2,
        expected_generation_observations=sum(
            sum(len(c[b]) for b in BANKS) * (1 + sum(r["case"] == c["case"] for r in requests))
            for c in data["cases"]
        ),
        expected_probe_vectors=sum(
            (1 + sum(r["case"] == c["case"] for r in requests))
            * sum(len(row["prefixes"]) for row in spec["rows"][str(c["case"])])
            for c in data["cases"]
        ),
        generation_banks=list(BANKS),
        runtime_limit_seconds=4800,
        comparison=(
            "Exact original50-step canonical controls must match history,exported weights,teacher "
            "and complete merged answers before any accepted-set fit. Accepted-set changes only "
            "the five initially accepted auxiliary rows per full-size case, using existing "
            "reference aliases and the original canonical-length denominator. The two auxiliary "
            "base/reference disagreements use identical canonical CE in both arms. Targets, "
            "supports,learning rates,orders,step count,clipping,scaling and optimizer are fixed."
        ),
        gradient_observer=(
            "Read-only cumulative-gradient snapshots after target and auxiliary backward; "
            "unscale,then subtract to measure auxiliary contribution before global clipping. "
            "Floating accumulation rounding remains. This is a mediator diagnostic,not "
            "an independent auxiliary-only backward or exact attribution to an EOS feature. "
            "Both canonical controls include the observer and must retain exact original histories."
        ),
        primary_causal_prediction=(
            "At least two of three representations,including selected paired neurons,halve "
            "case1 P(subscript2 | oxygen prompt,O) versus their canonical endpoint while retaining "
            "all4/4 transfer answers in both cases and preserving/restoring the oxygen answer O. "
            "This supports the auxiliary-loss hypothesis "
            "on these development cases,not practical adaptation superiority."
        ),
        practical_requirements=(
            "Report all complete-answer gains/losses,truncations,refusals,accepted mass and "
            "wrong-continuation competition. A useful candidate still needs8/8 transfers,no "
            "new genuine greedy preservation errors and the unchanged0.02 probability-drop "
            "target on eligible probes. No automatic win when controls fail or a proxy improves."
        ),
        limits=(
            "Two exposed counterfactual edits;known oxygen/Lakers probes stay out of training. "
            "No hard protection controller or new selector is tested. A shared benefit across "
            "LoRA and neuron patches is objective repair,not selection superiority. Existing "
            "canonical endpoints supply remaining matched references after two exact controls."
        ),
    )
    # These fields belonged to the completed no-fit qualification, not this study.
    for name in (
        "endpoints",
        "expected_endpoints",
        "expected_generations",
        "hypotheses",
        "decisions",
        "positive_methods",
        "negative",
    ):
        spec.pop(name, None)
    return spec


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "parent-proof", "probes", "probe-proof", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.parent, args.parent_proof, args.probes, args.probe_proof)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        value["model"], revision=value["revision"], local_files_only=True
    )
    data = json.loads((args.parent / "inputs.json").read_text())
    oxygen = next(
        r
        for c in data["cases"]
        if c["case"] == 1
        for r in c["utility_test"]
        if r["id"] == "utility-25"
    )
    example = chat_response(
        tokenizer, messages(oxygen), "O₂", max_length=value["max_sequence_tokens"]
    )
    path = example.input_ids[0, example.prompt_length :].tolist()
    assert len(path) == 3 and path[-1] == tokenizer.eos_token_id
    assert path[:1] in next(r["prefixes"] for r in value["rows"]["1"] if r["id"] == "utility-25")
    value["tracked_suffix"] = {"case": 1, "id": "utility-25", "prefix": path[:1], "token": path[1]}
    # Validate every alias without opening model weights or truncating any response.
    for case in data["cases"]:
        for row in case["preserve"]:
            for answer in row["answers"]:
                chat_response(
                    tokenizer, messages(row), answer, max_length=value["max_sequence_tokens"]
                )
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "new_fits": value["new_fits"],
                "generation_observations": value["expected_generation_observations"],
                "probe_vectors": value["expected_probe_vectors"],
                "eligible_auxiliary_rows": value["eligibility"],
            }
        )
    )


if __name__ == "__main__":
    main()
