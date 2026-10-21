"""Select a penalty from calibration only; report every generated comparison."""

import argparse
import json
import math
from pathlib import Path

from experiments.fine_transfer.analyze import prefix_hit
from experiments.fine_transfer.scope_analysis import scope_counts


def select_weight(fits, weights, cases, thresholds):
    """This interface receives no evaluation outputs or benchmark answer labels."""
    for weight in sorted(w for w in weights if w > 0):
        rows = [r for r in fits if r["weight"] == weight]
        if len(rows) != len(cases) or {r["case"] for r in rows} != set(cases):
            raise ValueError("Exactly one final calibration record per case required")
        if all(
            r["final"]["nll"] <= -math.log(thresholds["target_probability"])
            and r["final"]["max_answer_kl"] <= thresholds["max_answer_kl"]
            for r in rows
        ):
            return weight
    return None


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    completion, inputs = read("completion.json"), read("inputs.json")
    protocol = read("identity.json")["protocol"]
    assert completion["status"] == "complete" and completion["weights_restored"]
    fits = read("fits.json")
    assert len(fits) == 6
    expected = {(c["case"], w) for c in inputs["cases"] for w in protocol["preservation_weights"]}
    assert {(r["case"], r["weight"]) for r in fits} == expected
    selected = select_weight(
        fits,
        protocol["preservation_weights"],
        [c["case"] for c in inputs["cases"]],
        protocol["selection"],
    )
    assert selected == completion["selected_weight"]
    rows, details = [], []
    for case in inputs["cases"]:
        c = case["case"]
        base = read(f"case-{c}-base.json")
        for weight in protocol["preservation_weights"]:
            edited = read(f"case-{c}-w{int(weight)}.json")
            transfer = []
            for probe, output in zip(case["edit_probes"], edited["edit"], strict=True):
                transfer.append(
                    {"kind": probe["kind"], "hit": prefix_hit(output["text"], case["target"])}
                )
            counts = {}
            for name in ("calibration", "evaluation", "exposed_scope"):
                bank = {"new_target": case["target"], "probes": case[name]}
                count, decision = scope_counts(bank, base[name], edited[name])
                counts[name] = count
                details.extend({"case": c, "weight": weight, "bank": name, **r} for r in decision)
            losses = [
                i
                for i, (p, b, a) in enumerate(
                    zip(inputs["utility"], base["utility"], edited["utility"], strict=True)
                )
                if prefix_hit(b["text"], p["gold"]) and not prefix_hit(a["text"], p["gold"])
            ]
            rows.append(
                {
                    "case": c,
                    "weight": weight,
                    "transfer": transfer,
                    "scope": counts,
                    "utility_loss_indices": losses,
                }
            )
    groups = {}
    for weight in protocol["preservation_weights"]:
        chosen = [r for r in rows if r["weight"] == weight]
        groups[str(int(weight))] = {
            "original": sum(
                p["hit"] for r in chosen for p in r["transfer"] if p["kind"] == "rewrite"
            ),
            "untrained": sum(
                p["hit"]
                for r in chosen
                for p in r["transfer"]
                if p["kind"] in ("rephrase", "fresh_wording")
            ),
            "evaluation_leaks": sum(r["scope"]["evaluation"]["new_target_leakage"] for r in chosen),
            "exposed_scope_leaks": sum(
                r["scope"]["exposed_scope"]["new_target_leakage"] for r in chosen
            ),
            "calibration_leaks": sum(
                r["scope"]["calibration"]["new_target_leakage"] for r in chosen
            ),
            "evaluation_lost_references": [
                [r["case"], r["source_index"]]
                for r in details
                if r["weight"] == weight
                and r["bank"] == "evaluation"
                and r["lost_reference_prefix"]
            ],
            "evaluation_baseline_reference_count": sum(
                r["scope"]["evaluation"]["baseline_reference_prefix"] for r in chosen
            ),
            "utility_losses": [[r["case"], i] for r in chosen for i in r["utility_loss_indices"]],
        }
    checks = None
    if selected is not None:
        a, b = groups[str(int(selected))], groups["0"]
        checks = {
            "original": a["original"] == 2,
            "transfer": a["untrained"] >= 5 and a["untrained"] >= b["untrained"] - 1,
            "scope": a["evaluation_leaks"] <= 1
            and a["evaluation_leaks"] < b["evaluation_leaks"]
            and a["evaluation_leaks"] <= b["evaluation_leaks"] / 2,
            "reference_retention": not (
                set(map(tuple, a["evaluation_lost_references"]))
                - set(map(tuple, b["evaluation_lost_references"]))
            ),
            "utility_retention": not (
                set(map(tuple, a["utility_losses"])) - set(map(tuple, b["utility_losses"]))
            ),
        }
    return {
        "scope": "Two exposed development facts; six fits; frozen neuron supports",
        "selected_weight": selected,
        "advance_checks": checks,
        "advance_counts_pass": bool(checks) and all(checks.values()),
        "interpretation": (
            "Counts are not complete-answer quality; inspect all outputs. No selected "
            "weight is a calibration failure, not permission to select using evaluation "
            "results. An empty baseline-reference set cannot establish preservation."
        ),
        "groups": groups,
        "cases": rows,
    }, details


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, decisions = summarize(args.results)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("decisions.json", decisions)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
