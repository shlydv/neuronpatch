"""Build a full-output review queue and provisional exact-answer transitions.

Exact misses are not semantic errors. Review every response before making an
efficacy claim, including apparent exact hits and all truncated generations.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.paired_mlp.data import whole_hit
from experiments.relation_learning.select import choose


def summarize(rows):
    """Keep correction and damage separate, with explicit base denominators."""
    return {
        "count": len(rows),
        "base_exact": sum(r["base_exact"] for r in rows),
        "adapted_exact": sum(r["adapted_exact"] for r in rows),
        "corrections": sum(not r["base_exact"] and r["adapted_exact"] for r in rows),
        "regressions": sum(r["base_exact"] and not r["adapted_exact"] for r in rows),
        "base_truncated": sum(r["base"]["truncated"] for r in rows),
        "adapted_truncated": sum(r["adapted"]["truncated"] for r in rows),
    }


def assemble(case, base, fitted, *, condition, selected):
    rows = []
    groups = defaultdict(list)
    for bank, inputs in case.items():
        if not isinstance(inputs, list):
            continue
        expected = [r["id"] for r in inputs]
        if [r["id"] for r in base[bank]] != expected or [r["id"] for r in fitted[bank]] != expected:
            raise ValueError("Missing, duplicated or reordered predictions")
        for source, before, after in zip(inputs, base[bank], fitted[bank], strict=True):
            row = {
                "condition": condition,
                "selected": selected,
                "bank": bank,
                "id": source["id"],
                "family": source.get("family"),
                "world": source.get("world"),
                "variant": source.get("variant"),
                "evidence_condition": source.get("condition"),
                "messages": source.get("messages"),
                "prompt": source["prompt"],
                "references": source["answers"],
                "base": before,
                "adapted": after,
                "base_exact": whole_hit(before, source["answers"]),
                "adapted_exact": whole_hit(after, source["answers"]),
                "semantic_review": None,
            }
            rows.append(row)
            groups[(bank, "all", "all", "all")].append(row)
            if row["family"] is not None:
                groups[(bank, row["family"], row["variant"], row["evidence_condition"])].append(row)
    summary = [
        {
            "bank": key[0],
            "family": key[1],
            "variant": key[2],
            "evidence_condition": key[3],
            **summarize(value),
        }
        for key, value in groups.items()
    ]
    return rows, summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The selection file must already exist before this tool reads predictions.
    raw_choices = args.choices.read_bytes()
    choices = json.loads(raw_choices)
    selected = {r["id"] for r in choices["selected"]}
    fits = json.loads((args.results / "fits.json").read_text())
    expected = choose(
        [
            {
                "condition": r["condition"],
                "target_val_mean_nll": r["metrics"]["target_val_mean_nll"],
            }
            for r in fits
        ]
    )
    if choices["selected"] != expected:
        raise ValueError("Choices disagree with the validation-only rule")
    case = json.loads((args.results / "inputs.json").read_text())["cases"][0]
    base = json.loads((args.results / "case-0-base.json").read_text())
    if not selected.issubset({r["condition"]["id"] for r in fits}):
        raise ValueError("Selections do not match fits")
    queue, summaries = [], []
    for fit in fits:
        label = fit["condition"]["id"]
        fitted = json.loads((args.results / f"case-0-{label}.json").read_text())
        rows, summary = assemble(case, base, fitted, condition=label, selected=label in selected)
        queue.extend(rows)
        summaries.append({"condition": label, "selected": label in selected, "groups": summary})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "review_queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    (args.output / "provisional_exact.json").write_text(
        json.dumps(
            {
                "warning": (
                    "Exact scoring only; semantic review is pending. No efficacy conclusion."
                ),
                "choices_sha256": hashlib.sha256(raw_choices).hexdigest(),
                "fits": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
