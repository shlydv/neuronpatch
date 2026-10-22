"""Retrospective saved-update audit; no new fitting or model execution."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_checkpoints.geometry import nominal_update_norm
from experiments.paired_mlp.analysis import summarize


def audit(root):
    summary = summarize(root)
    fits = json.loads((root / "fits.json").read_text())
    inputs = [
        root / name for name in ("fits.json", "inputs.json", "identity.json", "completion.json")
    ]
    rows = []
    for fit in fits:
        case, condition = fit["case"], fit["condition"]
        name = condition["id"]
        adapter = root / f"case-{case}-{name}-adapter"
        if condition["family"] == "lora":
            inputs.extend(adapter / n for n in ("adapter_config.json", "adapter_model.safetensors"))
        else:
            inputs.extend(adapter / n for n in ("patch.json", "patch.safetensors"))
        inputs.extend(root / n for n in (f"case-{case}-base.json", f"case-{case}-{name}.json"))
        score = next(
            r for r in summary["cases"] if r["case"] == case and r["condition"]["id"] == name
        )
        rows.append(
            {
                "case": case,
                "condition": condition,
                "nominal_prequantization_update_l2": nominal_update_norm(
                    adapter, condition["family"]
                ),
                "target_test_hits": score["banks"]["target_test"]["hits"],
                "target_test_total": score["banks"]["target_test"]["total"],
                "scored_preservation_losses": sum(
                    len(score["banks"][b]["lost"]) for b in ("scope_test", "utility_test")
                ),
                "new_target_mentions": sum(
                    len(score["banks"][b]["new_target_mentions"])
                    for b in ("scope_test", "utility_test")
                ),
            }
        )
    paired_rates = []
    for low in rows:
        if low["condition"]["id"].endswith("-0"):
            high = next(
                r
                for r in rows
                if r["case"] == low["case"]
                and r["condition"]["id"] == low["condition"]["id"][:-1] + "1"
            )
            paired_rates.append(
                {
                    "case": low["case"],
                    "low": low["condition"]["id"],
                    "high": high["condition"]["id"],
                    "higher_lr_larger_norm": high["nominal_prequantization_update_l2"]
                    > low["nominal_prequantization_update_l2"],
                    "scored_loss_change": high["scored_preservation_losses"]
                    - low["scored_preservation_losses"],
                }
            )
    # Enumerate all such witnesses, rather than selecting only an attractive pair.
    # They refute perfect monotone separation of the frozen exact-score losses,
    # not every possible statistical use of update size.
    witnesses = [
        {
            "case": a["case"],
            "smaller_damaged": a["condition"]["id"],
            "smaller_l2": a["nominal_prequantization_update_l2"],
            "larger_scored_clean": b["condition"]["id"],
            "larger_l2": b["nominal_prequantization_update_l2"],
            "scored_losses": a["scored_preservation_losses"],
        }
        for a in rows
        for b in rows
        if a["case"] == b["case"]
        and a["scored_preservation_losses"] > 0
        and b["scored_preservation_losses"] == 0
        and a["nominal_prequantization_update_l2"] < b["nominal_prequantization_update_l2"]
    ]
    return {
        "scope": (
            "Retrospective analysis of all 24 existing fitted patches; "
            "zero new fits or model forwards."
        ),
        "norm_definition": (
            "Frobenius norm of nominal prequantization weight increments. "
            "Sparse deltas are direct; LoRA uses effective scaled BA, not raw factor norms. "
            "This is not a function-space distance or a relative base-weight norm."
        ),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(inputs))
        },
        "rows": rows,
        "within_support_learning_rate_pairs": paired_rates,
        "all_smaller_damaged_larger_clean_witnesses": witnesses,
        "limitations": [
            (
                "All examples and outcomes were already observed. This is exploratory, "
                "not a preregistered predictive validation."
            ),
            (
                "There are only two exposed facts; paraphrases and shared utility "
                "questions are correlated."
            ),
            (
                "Frozen exact-score losses include valid aliases and output-form defects. "
                "Read complete_v1/REPORT.md for genuine errors; do not describe each "
                "scored loss as a factual mistake."
            ),
            (
                "Learning rate, optimizer trajectory, norm and update direction change "
                "together; this analysis does not isolate their causal effects."
            ),
            (
                "Support-dependent model sensitivity is not measured by parameter-space "
                "norm alone. No universal safety threshold or better method is established."
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fits": len(result["rows"]),
                "rate_pairs": len(result["within_support_learning_rate_pairs"]),
                "larger_norm_at_high_rate": sum(
                    r["higher_lr_larger_norm"] for r in result["within_support_learning_rate_pairs"]
                ),
                "higher_scored_loss_at_high_rate": sum(
                    r["scored_loss_change"] > 0
                    for r in result["within_support_learning_rate_pairs"]
                ),
                "smaller_damaged_larger_clean_witnesses": len(
                    result["all_smaller_damaged_larger_clean_witnesses"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
