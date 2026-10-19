"""Compare complete paired FiNE runs without pooling unequal probe categories."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def finite_scores(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Missing score list")
    if not all(isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError("Invalid accuracy values")
    return mean(values)


def summarize(directory, cases):
    rows = []
    for i in range(cases):
        pair = [
            json.loads((directory / f"case-{i:03d}-random-{flag}.json").read_text())
            for flag in (False, True)
        ]
        for expected, result in zip((False, True), pair, strict=True):
            if result["case"] != i or result["random"] is not expected:
                raise ValueError("Condition metadata mismatch")
        for key in ("request", "probes", "baseline", "baseline_scores"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Unpaired {key} in case {i}")
        for result in pair:
            size = len(result["probes"])
            if size == 0 or any(
                len(result[key]) != size
                for key in ("baseline", "edited", "baseline_scores", "edited_scores")
            ):
                raise ValueError(f"Probe count mismatch in case {i}")
            by_kind = defaultdict(list)
            for probe, before, after in zip(
                result["probes"], result["baseline_scores"], result["edited_scores"], strict=True
            ):
                if before["target"] != after["target"]:
                    raise ValueError("Teacher-forced target changed across conditions")
                kind = probe["kind"]
                key = (
                    "locality_agreement"
                    if kind.startswith("locality/")
                    else "teacher_forced_token_accuracy"
                )
                by_kind[kind].append(finite_scores(after[key]))
            rows.append(
                {
                    "case": i,
                    "random": result["random"],
                    "categories": {
                        kind: {"value": mean(values), "probes": len(values)}
                        for kind, values in by_kind.items()
                    },
                }
            )
    categories = sorted({kind for row in rows for kind in row["categories"]})
    summary = {}
    for kind in categories:
        values = {}
        for flag, label in ((False, "selected"), (True, "random")):
            members = [
                r["categories"][kind]
                for r in rows
                if r["random"] is flag and kind in r["categories"]
            ]
            values[label] = {
                "case_mean": mean(r["value"] for r in members),
                "cases": len(members),
                "probes": sum(r["probes"] for r in members),
            }
        values["selected_minus_random"] = (
            values["selected"]["case_mean"] - values["random"]["case_mean"]
        )
        summary[kind] = values
    return {
        "cases": cases,
        "weighting": "mean probes within category per case, then mean cases",
        "missing_categories": "excluded for both conditions, counts reported",
        "scope": "exploratory paired subset; no full-benchmark reproduction or significance claim",
        "categories": summary,
        "per_case": rows,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text())
    if manifest["cases"] != args.cases:
        raise ValueError("Manifest case count differs")
    if args.cases <= 0:
        raise ValueError("Positive case count required")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.results, args.cases)
    result["manifest"] = manifest
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["categories"], indent=2))


if __name__ == "__main__":
    main()
