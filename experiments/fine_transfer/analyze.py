"""Summarize every pilot setting with separate transfer and preservation counts."""

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

from experiments.fine_reference.audit_generations import normalize, starts_with_answer


def prefix_hit(text, gold):
    """Use the reference name rule, but do not accept 4.5 or 4,000 as 4.

    The utility fixture introduces numeric answers absent from the two edited
    target names. This guard was added before reading any pilot answer outputs.
    As with the original diagnostic, a prefix is not full-response correctness.
    """
    if isinstance(gold, str) and gold.isdigit():
        text = normalize(text)
        match = re.match(r"[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
        if match is None:
            return False
        if match.end() < len(text) and (text[match.end()].isalnum() or text[match.end()] == "_"):
            return False
        return Decimal(match.group().replace(",", "")) == Decimal(gold)
    return starts_with_answer(text, gold)


def summarize(root):
    completion = json.loads((root / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["weights_restored"]
    assert len(completion["fits"]) == 8
    identity = json.loads((root / "identity.json").read_text())
    assert (
        completion["final_fingerprint"]
        == identity["fingerprint"]
        == identity["protocol"]["base_fingerprint"]
    )
    utility = json.loads((root / "base-utility.json").read_text())
    assert len(utility) == 32
    base_hits = {i for i, r in enumerate(utility) if prefix_hit(r["text"], r["gold"])}
    groups, details = {}, []
    conditions = [
        "fine-original",
        "fine_augmented",
        "gradient_single",
        "gradient_consensus",
        "consensus_layer_matched_random",
    ]
    for condition in conditions:
        counts = {
            "original": 0,
            "training": 0,
            "held_out_wording": 0,
            "held_out_wording_n": 0,
            "utility_retained": 0,
            "utility_baseline_correct": 2 * len(base_hits),
            "utility_losses": [],
        }
        for case in (0, 1):
            row = json.loads((root / f"case-{case}-{condition}.json").read_text())
            assert len(row["utility"]) == len(utility)
            for probe, before, after in zip(
                row["probes"], row["before"], row["after"], strict=True
            ):
                hit = prefix_hit(after["text"], probe["gold"])
                if probe["kind"] == "rewrite":
                    counts["original"] += hit
                if probe["kind"].startswith("train/"):
                    counts["training"] += hit
                if probe["kind"] in ("rephrase", "fresh_wording"):
                    counts["held_out_wording"] += hit
                    counts["held_out_wording_n"] += 1
                details.append(
                    {
                        "condition": condition,
                        "case": case,
                        **probe,
                        "before": before["text"],
                        "after": after["text"],
                        "prefix_hit": hit,
                        "truncated": after["truncated"],
                    }
                )
            for i in sorted(base_hits):
                hit = prefix_hit(row["utility"][i]["text"], utility[i]["gold"])
                counts["utility_retained"] += hit
                if not hit:
                    counts["utility_losses"].append([case, i])
        assert counts["held_out_wording_n"] == 6
        groups[condition] = counts
    candidate = groups["gradient_consensus"]
    losses = {tuple(x) for x in candidate["utility_losses"]}
    comparisons = {}
    for control in ("fine_augmented", "gradient_single"):
        other = groups[control]
        comparisons[control] = {
            "additional_transfer_prefixes": candidate["held_out_wording"]
            - other["held_out_wording"],
            "new_utility_losses": sorted(losses - {tuple(x) for x in other["utility_losses"]}),
        }
    gate = (
        candidate["original"] == 2
        and len(base_hits) >= 8
        and all(
            r["additional_transfer_prefixes"] >= 2 and not r["new_utility_losses"]
            for r in comparisons.values()
        )
    )
    return {
        "status": "Exposed two-case exploratory pilot; not a superiority claim",
        "baseline_utility_correct": len(base_hits),
        "groups": groups,
        "comparisons": comparisons,
        "frozen_advance_gate_passed": gate,
        "interpretation": (
            "Even a gate pass needs output review, identity value beyond matched random, "
            "fresh cases, low-rank baselines and compute accounting."
        ),
    }, details


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, details = summarize(args.results)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [("summary.json", summary), ("answer_details.json", details)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
