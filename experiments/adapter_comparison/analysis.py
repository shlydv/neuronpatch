"""Choose learning rates from calibration, then expose every development score."""

import argparse
import json
import math
from pathlib import Path

from experiments.fine_transfer.analyze import prefix_hit
from experiments.fine_transfer.scope_analysis import scope_counts


def choose_rates(fits, conditions, cases, thresholds):
    """Receive only training/calibration metrics, never generated evaluation text."""
    expected = {(case, row["id"]) for case in cases for row in conditions}
    keys = [(row["case"], row["condition"]) for row in fits]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Exactly one fit per case and condition required")
    if len({row["id"] for row in conditions}) != len(conditions):
        raise ValueError("Unique condition identifiers required")
    chosen = {}
    for family in sorted({row["family"] for row in conditions}):
        candidates = sorted(
            (row for row in conditions if row["family"] == family), key=lambda row: row["lr"]
        )
        chosen[family] = None
        for condition in candidates:
            metrics = [row["final"] for row in fits if row["condition"] == condition["id"]]
            if all(
                math.isfinite(row["nll"])
                and math.isfinite(row["max_answer_kl"])
                and row["nll"] <= -math.log(thresholds["target_probability"])
                and row["max_answer_kl"] <= thresholds["max_answer_kl"]
                for row in metrics
            ):
                chosen[family] = condition["id"]
                break
    return chosen


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    inputs, identity, completion, fits = [
        read(name) for name in ("inputs.json", "identity.json", "completion.json", "fits.json")
    ]
    protocol = identity["protocol"]
    selected = choose_rates(
        fits,
        protocol["conditions"],
        [row["case"] for row in inputs["cases"]],
        protocol["selection"],
    )
    assert completion["status"] == "complete" and completion["weights_restored"]
    assert completion["selected"] == selected
    assert completion["final_fingerprint"] == identity["fingerprint"]
    rows, decisions = [], []
    for case in inputs["cases"]:
        c = case["case"]
        base = read(f"case-{c}-base.json")
        for condition in protocol["conditions"]:
            name = condition["id"]
            output = read(f"case-{c}-{name}.json")
            scope = {}
            for bank in ("calibration", "evaluation", "exposed_scope"):
                scope[bank], detail = scope_counts(
                    {"new_target": case["target"], "probes": case[bank]}, base[bank], output[bank]
                )
                decisions.extend({"case": c, "condition": name, "bank": bank, **d} for d in detail)
            transfer = [
                {"kind": p["kind"], "hit": prefix_hit(a["text"], case["target"])}
                for p, a in zip(case["edit_probes"], output["edit"], strict=True)
            ]
            losses = [
                i
                for i, (p, b, a) in enumerate(
                    zip(inputs["utility"], base["utility"], output["utility"], strict=True)
                )
                if prefix_hit(b["text"], p["gold"]) and not prefix_hit(a["text"], p["gold"])
            ]
            rows.append(
                {
                    "case": c,
                    "condition": name,
                    "transfer": transfer,
                    "scope": scope,
                    "utility_loss_indices": losses,
                    "utility_exact_agreement": sum(
                        b == a for b, a in zip(base["utility"], output["utility"], strict=True)
                    ),
                }
            )
    groups = {}
    for condition in protocol["conditions"]:
        name = condition["id"]
        subset = [row for row in rows if row["condition"] == name]
        groups[name] = {
            "family": condition["family"],
            "lr": condition["lr"],
            "selected_from_calibration": selected[condition["family"]] == name,
            "original": sum(
                p["hit"] for row in subset for p in row["transfer"] if p["kind"] == "rewrite"
            ),
            "untrained": sum(
                p["hit"]
                for row in subset
                for p in row["transfer"]
                if p["kind"] in ("rephrase", "fresh_wording")
            ),
            **{
                f"{bank}_leaks": sum(row["scope"][bank]["new_target_leakage"] for row in subset)
                for bank in ("calibration", "evaluation", "exposed_scope")
            },
            "lost_evaluation_references": sum(
                row["scope"]["evaluation"]["lost_reference_prefix"] for row in subset
            ),
            "utility_losses": [
                [row["case"], i] for row in subset for i in row["utility_loss_indices"]
            ],
            "utility_exact_agreement": sum(row["utility_exact_agreement"] for row in subset),
        }
        group = groups[name]
        group["advance_counts_pass"] = (
            group["selected_from_calibration"]
            and group["original"] == 2
            and group["untrained"] >= 5
            and group["evaluation_leaks"] <= 1
            and group["lost_evaluation_references"] == 0
            and not group["utility_losses"]
        )
    return {
        "selected": selected,
        "groups": groups,
        "cases": rows,
        "full_output_review_complete": False,
        "interpretation": (
            "Exposed development cases, not fresh confirmation. Prefix counts cannot "
            "establish full-answer quality. No qualifying calibration setting means "
            "no selected candidate; report every fit without choosing from evaluation."
        ),
    }, decisions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, decisions = summarize(args.results)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("decisions.json", decisions)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"selected": summary["selected"], "groups": summary["groups"]}, indent=2))


if __name__ == "__main__":
    main()
