"""Freeze known positive/negative controls before collecting missing probabilities."""

import argparse
import json
from pathlib import Path

from experiments.paired_endpoint.prepare import prepare as parent_protocol
from experiments.paired_mlp.data import messages, whole_hit
from neuronpatch.supervision import chat_response


def token_spec(tokenizer, row, cap):
    """Finite aliases include known formal UK wording and a final period variant."""
    aliases = list(row["answers"])
    if "United Kingdom" in aliases:
        aliases += [
            "United Kingdom of Great Britain and Northern Ireland",
            "the United Kingdom of Great Britain and Northern Ireland",
        ]
    aliases = list(dict.fromkeys([*aliases, *[a + "." for a in aliases if not a.endswith(".")]]))
    paths, prompt = [], None
    for text in aliases:
        sample = chat_response(tokenizer, messages(row), text, max_length=cap)
        p = sample.input_ids[0, : sample.prompt_length].tolist()
        assert prompt is None or prompt == p
        prompt = p
        paths.append(sample.input_ids[0, sample.prompt_length :].tolist())
    prefixes = list(dict.fromkeys(tuple(p[:i]) for p in paths for i in range(len(p))))
    return {
        **row,
        "accepted_aliases": aliases,
        "paths": paths,
        "prompt_tokens": prompt,
        "prefixes": [list(p) for p in prefixes],
    }


def prepare(parent, integrity, tokenizer):
    spec = parent_protocol(parent, integrity)
    data = json.loads((parent / "inputs.json").read_text())
    spec["endpoints"] = [e for e in spec["endpoints"] if e["condition"]["lr"] == 0.0003]
    assert len(spec["endpoints"]) == 6
    banks = {}
    for case in data["cases"]:
        c = case["case"]
        baseline = json.loads((parent / f"case-{c}-base.json").read_text())
        ids = {r["id"] for r in case["preserve"]} | {"utility-25"}
        if c == 1:
            ids |= {"case-1-scope-test-2", "case-1-scope-test-3"}
        rows = []
        for bank in ("preserve", "scope_test", "utility_test"):
            for row, old in zip(case[bank], baseline[bank], strict=True):
                if row["id"] not in ids:
                    continue
                value = token_spec(tokenizer, row, spec["max_sequence_tokens"])
                value["source_bank"] = bank
                value["retention_eligible"] = whole_hit(old, row["answers"])
                value["role"] = (
                    "retention"
                    if value["retention_eligible"]
                    else "unresolved_reference_disagreement"
                )
                rows.append(value)
        assert {r["id"] for r in rows} == ids
        banks[str(c)] = rows
    spec.update(
        id="outcome-guard-qualification-v1-20261023",
        rows=banks,
        generation_banks=["target_test"],
        expected_generations=sum(
            4 * (len(banks[str(c["case"])]) + len(c["target_test"])) for c in data["cases"]
        ),
        maximum_absolute_mass_drop=0.02,
        positive_methods=["lora-43-0", "paired_random-11-0"],
        negative={
            "case": 1,
            "method": "paired_mean-43-0",
            "ids": ["case-1-scope-test-3", "utility-25"],
        },
        runtime_limit_seconds=1800,
        expected_endpoints=6,
        new_fits=0,
        hypotheses=[
            "All four successful LoRA/random controls retain accepted mass "
            "within0.02 on eligible rows.",
            "The selected case1 endpoint loses more than0.02 accepted mass "
            "on both Lakers and oxygen.",
        ],
        decisions=(
            "Require exact merged canonical-teacher and all selected generated-answer replay, "
            "all required positive endpoints admitted, both negative rows flagged by probability, "
            "and separate complete-answer validation. "
            "Any failed prerequisite blocks qualification. "
            "Collect full next-token logits at every unique finite-response prefix; "
            "do not estimate "
            "missing alias branches. No threshold adjustment or new fitting."
        ),
        limits=(
            "Known exposed calibration controls, not unbiased confirmation or prevention. "
            "A finite response set omits other valid strings. Base/reference disagreements are "
            "reported and excluded from retention gates, not silently corrected. "
            "Passing this test would license a trained comparison, not a robust library claim."
        ),
    )
    return spec


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    original = json.loads((args.parent / "identity.json").read_text())["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(
        original["model"], revision=original["revision"], local_files_only=True
    )
    spec = prepare(args.parent, args.integrity, tokenizer)
    with args.output.open("x") as stream:
        stream.write(json.dumps(spec, indent=2) + "\n")
    prefixes = sum(len(r["prefixes"]) for rows in spec["rows"].values() for r in rows)
    print(
        json.dumps(
            {
                "rows": sum(map(len, spec["rows"].values())),
                "prefixes_per_base_and_three_endpoints": prefixes,
                "generations": spec["expected_generations"],
                "new_fits": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
