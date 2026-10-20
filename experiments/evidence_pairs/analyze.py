"""Summarize frozen development comparisons; select with calibration only."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def group(row):
    if row["role"] == "invariant":
        return "invariant"
    return "supported" if row["kind"] == "answerable" else "missing"


def counts(rows):
    return {
        name: {
            "correct": sum(r["em"] for r in rows if group(r) == name),
            "n": sum(group(r) == name for r in rows),
        }
        for name in ("supported", "missing", "invariant")
    }


def preserve(rows, baseline, name=None):
    now = {r["id"]: r for r in rows}
    eligible = [r for r in baseline if r["em"] == 1 and (name is None or group(r) == name)]
    retained = sum(now[r["id"]]["em"] for r in eligible)
    return {
        "retained": retained,
        "n": len(eligible),
        "fraction": retained / len(eligible) if eligible else None,
    }


def discrimination(rows, signals):
    metadata = {r["id"]: r for r in rows}
    pos, neg, pairs = [], [], {}
    for row in signals:
        meta = metadata[row["id"]]
        if meta["role"] != "primary":
            continue
        score = row["unknown_log_odds"]
        (pos if meta["kind"] == "missing" else neg).append(score)
        pairs.setdefault(meta["pair_id"], {})[meta["kind"]] = score
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    gaps = [p["missing"] - p["answerable"] for p in pairs.values()]
    return {
        "unknown_log_odds_auroc": auc,
        "paired_positive_gaps": sum(g > 0 for g in gaps),
        "pairs": len(gaps),
        "mean_paired_gap": sum(gaps) / len(gaps),
        "note": "Fixed bias leaves AUROC and these gaps unchanged in real arithmetic.",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.results
    protocol, completion = read(root / "protocol.json"), read(root / "completion.json")
    base_cal = read(root / "base/cal_raw.json")
    base_cal_calibrated = read(root / "base/cal_calibrated.json")
    summary = {
        "completion": completion,
        "headroom": read(root / "headroom.json"),
        "scope": "controlled development, assistant-reviewed labels, one model/seed",
        "base_cal": counts(base_cal),
        "calibrated_base_cal": counts(base_cal_calibrated),
    }
    if not summary["headroom"]["proceed"]:
        summary["conclusion"] = "Stopped under frozen headroom rule; no fitted method tested."
        write(args.output / "summary.json", summary)
        return
    assert completion["status"] == "all frozen fits complete" and completion["base_restored"]
    base = read(root / "base/val_raw.json")
    calibrated_base = read(root / "base/val_calibrated.json")
    utility_base = read(root / "base/utility.json")
    summary.update(
        base=counts(base),
        calibrated_base=counts(calibrated_base),
        base_discrimination=discrimination(base, read(root / "base/val_signals.json")),
        base_utility={"correct": sum(r["em"] for r in utility_base), "n": len(utility_base)},
        fits=[],
        selected={},
    )
    changes = []
    for config in protocol["fits"]:
        directory = root / config["name"]
        cal = read(directory / "calibration.json")
        rows = read(directory / "val_calibrated.json")
        raw = read(directory / "val_raw.json")
        utility = read(directory / "utility.json")
        primary = counts(rows)
        retained = {k: preserve(rows, base, k) for k in ("supported", "invariant")}
        retained["utility"] = preserve(utility, utility_base)
        gain = primary["missing"]["correct"] / primary["missing"]["n"] - (
            summary["calibrated_base"]["missing"]["correct"]
            / summary["calibrated_base"]["missing"]["n"]
        )
        passes = gain >= protocol["success"][
            "minimum_val_missing_gain_over_calibrated_base"
        ] and all(
            retained[k]["fraction"] is not None and retained[k]["fraction"] >= 0.95
            for k in retained
        )
        item = {
            **config,
            "calibration": cal,
            "raw": counts(raw),
            "calibrated": primary,
            "utility": {"correct": sum(r["em"] for r in utility), "n": len(utility)},
            "retention": retained,
            "missing_gain_over_calibrated_base": gain,
            "numeric_development_gate_pass": passes,
            "discrimination": discrimination(rows, read(directory / "val_signals.json")),
            "fit_summary": read(directory / "fit_summary.json"),
        }
        history = read(directory / "history.json")
        assert len(history) == protocol["epochs"] * protocol["data_budget"]["train_pairs"]
        assert item["fit_summary"]["optimizer_skips"] == 0
        item["last_16_train_mean"] = {
            k: sum(r[k] for r in history[-16:]) / 16
            for k in ("primary_ce", "pair_loss", "control_ce", "control_max_kl")
        }
        summary["fits"].append(item)
        for old, new in zip(base, rows):
            assert old["id"] == new["id"]
            if old["generated_token_ids"] != new["generated_token_ids"]:
                changes.append(
                    {
                        "fit": config["name"],
                        "id": new["id"],
                        "pair_id": new["pair_id"],
                        "group": group(new),
                        "base": old["text"],
                        "edited": new["text"],
                        "answers": new["answers"],
                        "base_em": old["em"],
                        "edited_em": new["em"],
                    }
                )
    for family in ("guided", "random", "lora"):
        for objective in ("ce", "ce+pair"):
            options = [
                f for f in summary["fits"] if f["family"] == family and f["objective"] == objective
            ]
            eligible = [f for f in options if f["calibration"]["eligible"]]
            if not eligible:
                summary["selected"][f"{family}/{objective}"] = None
                continue
            best = max(
                eligible,
                key=lambda f: (
                    f["calibration"]["chosen"]["balanced_em"],
                    -f["lr"],
                    -f["calibration"]["chosen"]["bias"],
                ),
            )
            summary["selected"][f"{family}/{objective}"] = best["name"]
    write(args.output / "summary.json", summary)
    write(args.output / "changed_cases.json", changes)
    lines = [
        (
            "| Setting | Bias | Supported /64 | Missing /64 | Retained facts /128 | "
            "Utility /64 | AUROC | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for f in summary["fits"]:
        c = f["calibrated"]
        lines.append(
            f"| {f['name']} | {f['calibration']['chosen']['bias']} | "
            f"{c['supported']['correct']:g} | {c['missing']['correct']:g} | "
            f"{c['invariant']['correct']:g} | {f['utility']['correct']:g} | "
            f"{f['discrimination']['unknown_log_odds_auroc']:.3f} | "
            f"{'pass' if f['numeric_development_gate_pass'] else 'fail'} |"
        )
    (args.output / "table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
