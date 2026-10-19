"""Compare completed objective arms under identical neutral decoding."""

import hashlib
import json
from pathlib import Path

from analyze_coverage import aligned, transitions
from compare_predictions import compare
from data import metrics

ROOT = Path(__file__).parent


def main():
    coverage = json.loads((ROOT / "coverage_512/summary.json").read_text())
    summaries = {
        variant: json.loads((ROOT / f"token_loss_{variant}/summary.json").read_text())
        for variant in ("mean5", "max1")
    }
    if any(len(s["runs"]) != 2 for s in summaries.values()):
        raise ValueError("wait for all four objective fits to finish")
    base_rows = json.loads((ROOT / "decoding_512/base.json").read_text())
    base = {r["id"]: r for r in base_rows}
    known = {r["id"] for r in base_rows if r["kind"] == "answerable" and r["em"] == 1}
    result = {
        "base_metrics": metrics(base_rows),
        "neutral_base_exact_correct": len(known),
        "scope": "Development comparisons, same checkpoints' explicit neutral decoder",
        "runs": {},
        "max_vs_mean5": {},
    }
    by_rate = {}
    for variant, summary in summaries.items():
        for key in (
            "model",
            "revision",
            "data_sha256",
            "anchors",
            "anchor_order",
            "target_modules",
            "replay_exposures_per_fit",
            "steps_per_epoch",
            "epochs",
        ):
            if summary[key] != coverage[key]:
                raise ValueError(f"control metadata differs: {variant}/{key}")
        assert summary["effective_generation_config"]["repetition_penalty"] == 1.0
        assert summary["effective_generation_config"]["do_sample"] is False
        assert summary["token_reduction"] == ("max" if variant == "max1" else "mean")
        for name, run in summary["runs"].items():
            assert run["optimizer_steps_total"] == 256
            assert run["kl_coefficient"] == (5.0 if variant == "mean5" else 1.0)
            reference_name = f"coverage_512-lora-lr{run['lr']}-kl1.0-epoch1"
            control_path = ROOT / "decoding_512" / f"{reference_name}.json"
            path = ROOT / f"token_loss_{variant}" / f"{name}-val.json"
            control_rows = json.loads(control_path.read_text())
            rows = json.loads(path.read_text())
            old, new = aligned(control_rows, rows)
            assert new.keys() == base.keys()
            by_rate[(variant, run["lr"])] = rows
            key = f"{variant}/{name}"
            result["runs"][key] = {
                "control_metrics": metrics(control_rows),
                "metrics": metrics(rows),
                "paired_article_uncertainty": compare(control_rows, rows),
                "base_correct_transitions": transitions(old, new, known),
                "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "control_prediction_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
                "training_seconds": run["training_seconds"],
                "review_changes": [
                    {
                        "id": i,
                        "kind": new[i]["kind"],
                        "context": new[i]["context"],
                        "question": new[i]["question"],
                        "gold": new[i]["answers"],
                        "before": old[i]["output"],
                        "after": new[i]["output"],
                    }
                    for i in new
                    if old[i]["output"] != new[i]["output"]
                ],
            }
            print(
                key, result["runs"][key]["metrics"], result["runs"][key]["base_correct_transitions"]
            )
    for lr in (0.0001, 0.0003):
        result["max_vs_mean5"][str(lr)] = compare(by_rate[("mean5", lr)], by_rate[("max1", lr)])
    out = ROOT / "token_loss_comparison"
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
