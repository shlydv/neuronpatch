"""Expose every frozen intervention's correction and preservation outcomes."""

import argparse
import json
import math
from pathlib import Path

from experiments.fine_transfer.analyze import prefix_hit
from experiments.fine_transfer.scope_analysis import scope_counts


def summarize(root, *, partial=False):
    def read(name):
        return json.loads((root / name).read_text())

    inputs, identity, interventions, bases = [
        read(n)
        for n in (
            "inputs.json",
            "identity.json",
            "interventions.json",
            "base-metrics.json",
        )
    ]
    if partial:
        assert not (root / "completion.json").exists()
    else:
        completion = read("completion.json")
        assert completion["status"] == "complete" and completion["controls_exact"]
    protocol = identity["protocol"]
    criteria = protocol["criteria"]
    rows, decisions = [], []
    for case in inputs["cases"]:
        c = case["case"]
        base = read(f"case-{c}-base.json")
        base_nll = next(r for r in bases if r["case"] == c)["metrics"]["nll"]
        full_nll = next(r for r in interventions if r["case"] == c and r["condition"] == "full")[
            "metrics"
        ]["nll"]
        for condition in protocol["conditions"]:
            name = condition["id"]
            fit = next(
                (r for r in interventions if r["case"] == c and r["condition"] == name), None
            )
            if partial and fit is None:
                continue
            assert fit is not None
            output = read(f"case-{c}-{name}.json")
            scope = {}
            for bank in ("calibration", "evaluation", "exposed_scope"):
                scope[bank], detail = scope_counts(
                    {"new_target": case["target"], "probes": case[bank]}, base[bank], output[bank]
                )
                decisions += [{"case": c, "condition": name, "bank": bank, **d} for d in detail]
            transfer = [
                {"kind": p["kind"], "hit": prefix_hit(a["text"], case["target"])}
                for p, a in zip(case["edit_probes"], output["edit"], strict=True)
            ]
            lost = [
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
                    "metrics": fit["metrics"],
                    "retained_factor_scalars": fit["retained_factor_scalars"],
                    "fraction_of_full_target_nll_improvement": (
                        (base_nll - fit["metrics"]["nll"]) / (base_nll - full_nll)
                        if base_nll > full_nll
                        else None
                    ),
                    "transfer": transfer,
                    "scope": scope,
                    "utility_loss_indices": lost,
                    "utility_exact_agreement": sum(
                        b == a for b, a in zip(base["utility"], output["utility"], strict=True)
                    ),
                }
            )
    groups = {}
    for condition in protocol["conditions"]:
        name = condition["id"]
        subset = [r for r in rows if r["condition"] == name]
        eligible = len(subset) == len(inputs["cases"]) and all(
            r["metrics"]["nll"] <= -math.log(criteria["target_probability"])
            and r["metrics"]["max_answer_kl"] <= criteria["max_answer_kl"]
            for r in subset
        )
        groups[name] = {
            "available_cases": len(subset),
            "complete_condition": len(subset) == len(inputs["cases"]),
            "training_calibration_bounds_pass": eligible,
            "original": sum(
                p["hit"] for r in subset for p in r["transfer"] if p["kind"] == "rewrite"
            ),
            "untrained": sum(
                p["hit"]
                for r in subset
                for p in r["transfer"]
                if p["kind"] in ("rephrase", "fresh_wording")
            ),
            **{
                f"{bank}_leaks": sum(r["scope"][bank]["new_target_leakage"] for r in subset)
                for bank in ("calibration", "evaluation", "exposed_scope")
            },
            "lost_evaluation_references": sum(
                r["scope"]["evaluation"]["lost_reference_prefix"] for r in subset
            ),
            "utility_losses": [[r["case"], i] for r in subset for i in r["utility_loss_indices"]],
            "utility_exact_agreement": sum(r["utility_exact_agreement"] for r in subset),
            "untrained_denominator": sum(
                p["kind"] in ("rephrase", "fresh_wording") for r in subset for p in r["transfer"]
            ),
            "evaluation_denominator": sum(r["scope"]["evaluation"]["probes"] for r in subset),
        }
        g = groups[name]
        g["conditional_count_pass"] = (
            g["complete_condition"]
            and eligible
            and g["original"] == len(inputs["cases"])
            and g["untrained"] >= criteria["minimum_untrained_prefixes"]
            and g["evaluation_leaks"] <= criteria["maximum_evaluation_leaks"]
            and g["lost_evaluation_references"] == 0
            and not g["utility_losses"]
        )
    return {
        "groups": groups,
        "cases": rows,
        "full_output_review_complete": False,
        "partial": partial,
        "interpretation": (
            "Frozen co-trained adapter groups on exposed cases. No refit, best-setting "
            "selection, unique circuit or quality claim."
        ),
    }, decisions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    summary, decisions = summarize(args.results, partial=args.partial)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("decisions.json", decisions)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
