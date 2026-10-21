"""Supplement frozen prefix scores with coverage and literal target mentions.

These diagnostics are exploratory, never inputs to adapter/rate selection.
Exact string mentions are not semantic factuality or causal attribution labels.
"""

import argparse
import json
from pathlib import Path

from experiments.fine_transfer.analyze import prefix_hit


def inspect(root):
    def read(name):
        return json.loads((root / name).read_text())

    inputs = read("inputs.json")
    conditions = read("identity.json")["protocol"]["conditions"]
    rows = []
    for case in inputs["cases"]:
        c = case["case"]
        base = read(f"case-{c}-base.json")
        baseline_hits = [
            i
            for i, (probe, answer) in enumerate(
                zip(inputs["utility"], base["utility"], strict=True)
            )
            if prefix_hit(answer["text"], probe["gold"])
        ]
        for condition in conditions:
            name = condition["id"]
            output = read(f"case-{c}-{name}.json")
            mentions = {}
            target = case["target"].casefold()
            for bank in ("calibration", "evaluation", "exposed_scope", "utility"):
                mentions[bank] = [
                    i
                    for i, (old, new) in enumerate(zip(base[bank], output[bank], strict=True))
                    if target in new["text"].casefold() and target not in old["text"].casefold()
                ]
            rows.append(
                {
                    "case": c,
                    "condition": name,
                    "utility_baseline_hit_indices": baseline_hits,
                    "utility_edited_hit_indices": [
                        i
                        for i, (probe, answer) in enumerate(
                            zip(inputs["utility"], output["utility"], strict=True)
                        )
                        if prefix_hit(answer["text"], probe["gold"])
                    ],
                    "new_literal_target_mention_indices": mentions,
                }
            )
    return {
        "status": "Post-hoc diagnostics; frozen primary scoring and selection unchanged",
        "limitations": (
            "Literal case-insensitive target substrings anywhere in the capped continuation, "
            "absent from its paired base output. Not semantic aliases, correct attribution, "
            "or proof of factual damage. Utility prefix hits are not whole-answer quality."
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.results)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['rows'])} case/condition diagnostics")


if __name__ == "__main__":
    main()
