"""Report all early/full endpoints and measured prefix costs;do not retune the gate."""

import argparse
import json
from pathlib import Path

from experiments.paired_mlp.data import whole_hit


def summarize(original, results):
    protocol = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    assert completion["status"] == "complete"
    cases = {c["case"]: c for c in json.loads((original / "inputs.json").read_text())["cases"]}
    fits = json.loads((results / "fits.json").read_text())
    assert len(fits) == completion["new_fits"] == protocol["new_fits"]
    rows = []
    for fit in fits:
        assert (
            fit["final_history_exact"]
            and fit["final_parameters_exact"]
            and fit["final_live_teacher_exact"]
        )
        c, name = fit["case"], fit["condition"]["id"]
        case = cases[c]
        base = json.loads((original / f"case-{c}-base.json").read_text())
        states = {}
        for state, record in fit["states"].items():
            generated = json.loads((results / f"case-{c}-{name}-{state}.json").read_text())
            banks = {}
            for bank in protocol["generation_banks"]:
                lost = []
                mentions = []
                hits = 0
                for row, b, a in zip(case[bank], base[bank], generated[bank], strict=True):
                    hit = whole_hit(a, row["answers"])
                    hits += hit
                    if whole_hit(b, row["answers"]) and not hit:
                        lost.append(row["id"])
                    if (
                        case["target"].casefold() in a["text"].casefold()
                        and case["target"].casefold() not in b["text"].casefold()
                    ):
                        mentions.append(row["id"])
                banks[bank] = {
                    "hits": hits,
                    "total": len(case[bank]),
                    "scored_lost": lost,
                    "new_target_mentions": mentions,
                    "truncated": sum(a["truncated"] for a in generated[bank]),
                }
            states[state] = {
                "step": record["step"],
                "banks": banks,
                "merged_calibration": record["merged_calibration"],
                "nominal_prequantization_update_l2": record["nominal_prequantization_update_l2"],
            }
        common = fit["attach_seconds"] + fit["discovery_seconds"]
        snapshots = fit["checkpoints"]
        final_check = snapshots[-1]["monitoring_seconds"] - (
            snapshots[-2]["monitoring_seconds"] if len(snapshots) > 1 else 0
        )
        full_cost = (
            common + fit["training_seconds"] + final_check + fit["final_serialization_seconds"]
        )
        early_cost = None
        if fit["selected_step"] is not None:
            selected = next(s for s in snapshots if s["step"] == fit["selected_step"])
            early_cost = (
                common
                + selected["training_seconds"]
                + selected["monitoring_seconds"]
                + selected["serialization_seconds"]
            )
        rows.append(
            {
                "case": c,
                "condition": fit["condition"],
                "role": fit["role"],
                "states": states,
                "selected_step": fit["selected_step"],
                "observed_prefix_training_check_export_seconds": early_cost,
                "observed_fixed_training_one_check_export_seconds": full_cost,
                "discovery_seconds": fit["discovery_seconds"],
                "diagnostic_seconds": fit["diagnostic_seconds"],
                "mode_comparisons": {
                    str(s["step"]): s["mode_comparison"]
                    for s in snapshots
                    if "mode_comparison" in s
                },
            }
        )
    aggregate = {}
    for name in dict.fromkeys(r["condition"]["id"] for r in rows if r["role"] == "comparison"):
        group = [r for r in rows if r["condition"]["id"] == name and r["role"] == "comparison"]
        assert len(group) == 2
        aggregate[name] = {
            "selected_steps": [r["selected_step"] for r in group],
            "sum_discovery_seconds": sum(r["discovery_seconds"] for r in group),
            "sum_prefix_seconds": sum(
                r["observed_prefix_training_check_export_seconds"] for r in group
            )
            if all(r["selected_step"] is not None for r in group)
            else None,
            "sum_fixed_reference_seconds": sum(
                r["observed_fixed_training_one_check_export_seconds"] for r in group
            ),
            "states": {},
        }
        for state in ("early", "full"):
            if not all(state in r["states"] for r in group):
                continue
            aggregate[name]["states"][state] = {
                bank: {
                    "hits": sum(r["states"][state]["banks"][bank]["hits"] for r in group),
                    "total": sum(r["states"][state]["banks"][bank]["total"] for r in group),
                    "scored_lost": sum(
                        len(r["states"][state]["banks"][bank]["scored_lost"]) for r in group
                    ),
                    "new_target_mentions": sum(
                        len(r["states"][state]["banks"][bank]["new_target_mentions"]) for r in group
                    ),
                }
                for bank in protocol["generation_banks"]
            }
    return {
        "fits": len(fits),
        "observations": completion["observations"],
        "conditions": aggregate,
        "cases": rows,
        "full_output_review_complete": False,
        "cost_scope": (
            "Observed adaptation prefix includes attachment,discovery,all prior teacher checks/"
            "raw checkpoint writes,and selected export. Fixed reference counts training,one final "
            "check and export. Excludes common model load,deployment scoring,and diagnostics. "
            "The continued control suffix is not an actual early-stopped run. "
            "No production latency or replicated timing claim."
        ),
        "interpretation": (
            "Two exposed development facts and one deliberately strong random seed. "
            "No fresh confirmation or method-superiority conclusion. Original labels remain "
            "frozen;valid aliases and ambiguous questions need qualitative review."
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("original", "results", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.original, args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("cases", "conditions")}, indent=2)
    )


if __name__ == "__main__":
    main()
