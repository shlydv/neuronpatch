"""Evaluate the prospective predictor criteria without choosing a support or scale."""

import argparse
import json
from pathlib import Path

from experiments.paired_mlp.data import whole_hit


def analyze(parent, results):
    protocol = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    assert completion["status"] == "complete"
    records = json.loads((results / "records.json").read_text())
    data = json.loads((parent / "inputs.json").read_text())["cases"]
    cases, calibration = [], []
    for case in data:
        c = case["case"]
        primary = {
            r["condition"]: r
            for r in records
            if r["case"] == c and r["scale"] == protocol["generation_scale"]
        }
        baseline = primary["adam_gain"]["metrics"]
        candidate = primary["gain_risk"]["metrics"]
        bg, cg = baseline["task_nll_improvement"], candidate["task_nll_improvement"]
        bk, ck = baseline["heldout_mean_anchored_kl"], candidate["heldout_mean_anchored_kl"]
        informative = bg > 0 and bk >= 1e-8
        efficiency_ratio = (
            ((cg / bg) ** 2 * bk / max(ck, 1e-12)) if informative and cg > 0 else None
        )
        conditions = {
            "informative": informative,
            "positive_correction_gain": cg >= 0.001,
            "retains_half_baseline_gain": informative and cg >= 0.5 * bg,
            "twenty_percent_less_heldout_kl": informative and ck <= 0.8 * bk,
            "twenty_five_percent_more_gain_squared_per_kl": efficiency_ratio is not None
            and efficiency_ratio >= 1.25,
        }
        cases.append(
            {
                "case": c,
                "candidate_gain": cg,
                "baseline_gain": bg,
                "candidate_heldout_kl": ck,
                "baseline_heldout_kl": bk,
                "gain_squared_per_kl_ratio": efficiency_ratio,
                "candidate_kl_floor_applied": ck < 1e-12,
                "criteria": conditions,
                "passes_primary": all(conditions.values()),
            }
        )
        small = []
        for record in [
            r for r in records if r["case"] == c and r["scale"] == protocol["scales"][0]
        ]:
            observed = record["metrics"]["calibration_mean_anchored_kl"]
            predicted = record["predicted_calibration_mean_anchored_kl"]
            ratio = predicted / observed if observed >= 1e-8 else None
            small.append(
                {
                    "condition": record["condition"],
                    "observed": observed,
                    "predicted": predicted,
                    "ratio": ratio,
                    "passes": None if ratio is None else 0.25 <= ratio <= 4,
                }
            )
        informative_small = [r for r in small if r["ratio"] is not None]
        calibration.append(
            {
                "case": c,
                "conditions": small,
                "passes": bool(informative_small) and all(r["passes"] for r in informative_small),
            }
        )
    scored = []
    for case in data:
        c = case["case"]
        baseline = json.loads((parent / f"case-{c}-base.json").read_text())
        for record in [r for r in records if r["case"] == c and r["generated"]]:
            outputs = json.loads((results / f"{record['prefix']}-generations.json").read_text())
            banks = {}
            for bank in protocol["generation_banks"]:
                hits, lost, mentions = 0, [], []
                for row, before, after in zip(
                    case[bank], baseline[bank], outputs[bank], strict=True
                ):
                    hit = whole_hit(after, row["answers"])
                    hits += hit
                    if whole_hit(before, row["answers"]) and not hit:
                        lost.append(row["id"])
                    if (
                        case["target"].casefold() in after["text"].casefold()
                        and case["target"].casefold() not in before["text"].casefold()
                    ):
                        mentions.append(row["id"])
                banks[bank] = {
                    "hits": hits,
                    "total": len(case[bank]),
                    "scored_lost": lost,
                    "new_target_mentions": mentions,
                    "truncated": sum(r["truncated"] for r in outputs[bank]),
                }
            scored.append(
                {
                    "case": c,
                    "condition": record["condition"],
                    "scale": record["scale"],
                    "banks": banks,
                }
            )
    return {
        "one_step_edits": len(records),
        "observations": completion["observations"],
        "primary": cases,
        "small_step_calibration": calibration,
        "licenses_trained_method_pilot": all(r["passes_primary"] for r in cases)
        and all(r["passes"] for r in calibration),
        "generations": scored,
        "records": records,
        "full_output_review_complete": False,
        "scope": (
            "Two exposed facts and data-informed one-step edits. Predictor calibration only; "
            "no multi-step recipe,novel method or superiority established. "
            "Complete answer review remains separate."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    assert json.loads(args.integrity.read_text())["complete"]
    result = analyze(args.parent, args.results)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("records", "generations")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
