"""Report the no-training dose/bias diagnostic without selecting on validation."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.protected_rows.report import compare, indexed, split


def read(path):
    return json.loads(path.read_text())


def shift_groups(base, edited):
    baseline = indexed(base)
    return {
        kind: sum(
            row["unknown_log_odds"] - baseline[row["id"]]["unknown_log_odds"]
            for row in edited
            if row["kind"] == kind
        )
        / sum(row["kind"] == kind for row in edited)
        for kind in sorted({row["kind"] for row in edited})
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.diagnostic / "manifest.json")
    if not manifest.get("completed"):
        raise ValueError("The diagnostic is incomplete")
    for name in ("protected_full", "unprotected_full"):
        for part, expected in (("val", 128), ("utility", 32)):
            value = manifest["reproduction"][f"{name}/{part}"]
            if value["n"] != expected or value["mismatched_ids"]:
                raise ValueError("Exported generations do not reproduce")
    base = read(args.pilot / "base_val.json")
    utility = read(args.pilot / "base_utility.json")
    audit = read(args.audit)
    if (
        hashlib.sha256((args.pilot / "base_val.json").read_bytes()).hexdigest()
        != audit["baseline_predictions_sha256"]
    ):
        raise ValueError("Baseline-only audit belongs to different predictions")
    conditions = manifest["protocol"]["controls"]
    baseline_signals = read(args.diagnostic / "base_val_signals.json")
    predictions = {}
    results = {}
    for name in conditions:
        rows = read(args.diagnostic / name / "val.json")
        predictions[name] = rows
        lookup = indexed(rows)
        result = {
            "supported": compare(split(base, "answerable"), split(rows, "answerable")),
            "missing": compare(split(base, "missing"), split(rows, "missing")),
            "utility": compare(utility, read(args.diagnostic / name / "utility.json")),
        }
        cases = audit["cases"]
        correct_content = [row for row in cases if row["assessment"] == "correct_content"]
        true_missing_errors = [
            row for row in cases if row["kind"] == "missing" and row["assessment"] == "real_error"
        ]
        result["secondary_assistant_audit"] = {
            "scope": "Diagnostic subset; not independent human gold or complete relabeling",
            "correct_content_baseline_n": len(correct_content),
            "new_strict_refusal_ids": [
                row["id"] for row in correct_content if lookup[row["id"]]["abstained"]
            ],
            "true_missing_error_n": len(true_missing_errors),
            "true_missing_fixed_ids": [
                row["id"] for row in true_missing_errors if lookup[row["id"]]["abstained"]
            ],
        }
        result["mean_validation_log_odds_shift_by_kind"] = (
            {kind: manifest["protected_calibration_shift"] for kind in ("answerable", "missing")}
            if name == "unknown_bias"
            else shift_groups(baseline_signals, read(args.diagnostic / name / "val_signals.json"))
        )
        result["promotion_primary"] = (
            result["missing"]["em"] - sum(row["em"] for row in split(base, "missing")) / 64 >= 0.05
            and result["supported"]["retention"] >= 0.95
            and result["utility"]["retention"] >= 0.95
        )
        results[name] = result
    pairwise = {
        other: {
            kind: compare(
                split(predictions[other], kind), split(predictions["protected_full"], kind)
            )
            for kind in ("answerable", "missing")
        }
        for other in ("unprotected_matched", "unknown_bias")
    }
    record = {
        "scope": "Exploratory exposed-pilot diagnostic; no training or fresh confirmation",
        "manifest": manifest,
        "dose_match_adequate": manifest["relative_matching_error"] <= 0.1,
        "metrics": results,
        "protected_versus_controls": pairwise,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    lines = [
        "# No-training dose and refusal-bias diagnostic",
        "",
        record["scope"] + ".",
        "",
        "Calibration protected mean UNKNOWN log-odds shift: "
        f"{manifest['protected_calibration_shift']:.5f}.",
        f"Matched unprotected scale: {manifest['matched_scale']:.5f}; relative signal mismatch: "
        f"{manifest['relative_matching_error']:.2%}.",
        "Cold-loaded saved patches exactly reproduce all 320 validation/utility trajectories.",
        "",
        "| Condition | Supported /64 | Missing /64 | Old correct retained /38 | "
        "Utility /32 | Primary pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, values in results.items():
        lines.append(
            f"| {name} | {round(values['supported']['em'] * 64)} | "
            f"{round(values['missing']['em'] * 64)} | {values['supported']['retained']} | "
            f"{round(values['utility']['em'] * 32)} | {values['promotion_primary']} |"
        )
    lines += [
        "",
        "The mean matches one refusal signal on calibration, not the full output distribution. "
        "The constant-bias control only changes the first generated token. "
        "Secondary assistant labels are diagnostic and do not alter the frozen evaluator.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
