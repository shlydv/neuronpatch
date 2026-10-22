"""Retrospective label/answer leakage audit; does not change frozen scores."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.data import normalize, whole_hit


def audit(parent, summary):
    data = json.loads((parent / "inputs.json").read_text())
    cases = {c["case"]: c for c in data["cases"]}
    conditions = json.loads((parent / "identity.json").read_text())["protocol"]["conditions"]
    rows, anchors = [], []
    for c, case in cases.items():
        base = json.loads((parent / f"case-{c}-base.json").read_text())
        not_correct = [
            r["id"]
            for r, a in zip(case["preserve"], base["preserve"], strict=True)
            if not whole_hit(a, r["answers"])
        ]
        anchors.append(
            {
                "case": c,
                "preservation_examples": len(case["preserve"]),
                "baseline_reference_correct": len(case["preserve"]) - len(not_correct),
                "baseline_not_reference_correct_ids": not_correct,
            }
        )
        target_answers = {normalize(a) for r in case["train"] for a in r["answers"]}
        anchor_answers = {normalize(a) for r in case["preserve"] for a in r["answers"]}
        for condition in conditions:
            name = condition["id"]
            after = json.loads((parent / f"case-{c}-{name}.json").read_text())
            banks = {}
            for bank in ("scope_test", "utility_test", "scope_exposed"):
                failed = []
                for probe, b, a in zip(case[bank], base[bank], after[bank], strict=True):
                    if whole_hit(b, probe["answers"]) and not whole_hit(a, probe["answers"]):
                        text = normalize(a["text"])
                        failed.append(
                            {
                                "id": probe["id"],
                                "exact_task_training_answer": text in target_answers,
                                "exact_preservation_training_answer": text in anchor_answers,
                                "new_literal_target_mention": (
                                    case["target"].casefold() in a["text"].casefold()
                                    and case["target"].casefold() not in b["text"].casefold()
                                ),
                                "truncated": a["truncated"],
                            }
                        )
                banks[bank] = {
                    "scored_lost": len(failed),
                    "to_task_training_answer": sum(r["exact_task_training_answer"] for r in failed),
                    "to_preservation_training_answer": sum(
                        r["exact_preservation_training_answer"] for r in failed
                    ),
                    "rows": failed,
                }
                if bank != "scope_exposed":
                    reference = summary["conditions"][name][bank]
                    assert len(failed) <= reference["lost"]
            rows.append({"case": c, "condition": name, "banks": banks})
    for name in (r["id"] for r in conditions):
        for bank in ("scope_test", "utility_test"):
            assert (
                sum(r["banks"][bank]["scored_lost"] for r in rows if r["condition"] == name)
                == summary["conditions"][name][bank]["lost"]
            )
    return {
        "status": (
            "Retrospective objective diagnosis; all frozen scoring and method choices unchanged"
        ),
        "source_inputs_sha256": hashlib.sha256((parent / "inputs.json").read_bytes()).hexdigest(),
        "preservation_training_coverage": anchors,
        "conditions": rows,
        "interpretation": (
            "A preservation reference that the base does not already produce is an auxiliary "
            "supervised correction or wording constraint,not purely behavior retention. "
            "A damaged answer matching a training label is consistent with label spillover but "
            "does not causally identify which example or parameter caused it."
        ),
        "limits": (
            "Strict exact-answer losses include valid aliases and output-form differences. "
            "Counts are observations,not independent examples. Base answer/reference correctness "
            "requires semantic review. Label matches are exact normalized answers,"
            "not substring matches."
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.parent, json.loads(args.summary.read_text()))
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    main()
