"""Keep reference retention, answer leakage and continuation changes separate."""

import argparse
import hashlib
import json
from pathlib import Path

from .analyze import prefix_hit

CONDITIONS = (
    "fine-original",
    "fine_augmented",
    "gradient_single",
    "gradient_consensus",
    "consensus_layer_matched_random",
)


def digest_output(row):
    """Compare exact saved continuations without putting their text in fixtures."""
    return {
        "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
        "tokens_sha256": hashlib.sha256(json.dumps(row["tokens"]).encode()).hexdigest(),
        "truncated": row["truncated"],
    }


def scope_counts(bank, before, after):
    decisions = []
    for probe, base, edited in zip(bank["probes"], before, after, strict=True):
        base_reference = prefix_hit(base["text"], probe["reference_answers"])
        edited_reference = prefix_hit(edited["text"], probe["reference_answers"])
        base_target = prefix_hit(base["text"], bank["new_target"])
        edited_target = prefix_hit(edited["text"], bank["new_target"])
        decisions.append(
            {
                "source_index": probe["source_index"],
                "baseline_reference_prefix": base_reference,
                "edited_reference_prefix": edited_reference,
                "lost_reference_prefix": base_reference and not edited_reference,
                "baseline_new_target_prefix": base_target,
                "edited_new_target_prefix": edited_target,
                "new_target_leakage": edited_target and not base_target,
                "exact_continuation_agreement": base == edited,
            }
        )
    counts = {
        "probes": len(decisions),
        **{
            name: sum(row[name] for row in decisions)
            for name in (
                "baseline_reference_prefix",
                "edited_reference_prefix",
                "lost_reference_prefix",
                "baseline_new_target_prefix",
                "edited_new_target_prefix",
                "new_target_leakage",
                "exact_continuation_agreement",
            )
        },
    }
    counts["retained_baseline_reference_prefix"] = (
        counts["baseline_reference_prefix"] - counts["lost_reference_prefix"]
    )
    return counts, decisions


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    completion = read("completion.json")
    assert completion["status"] == "complete" and completion["new_fits"] == 0
    assert completion["cold_checks_before_new_probes"] and completion["weights_restored"]
    assert completion["final_fingerprint"] == read("identity.json")["fingerprint"]
    actual = [(r["case"], r["condition"]) for r in completion["conditions"]]
    expected = {(case, condition) for case in (0, 1) for condition in CONDITIONS}
    assert len(actual) == 10 and set(actual) == expected
    cases, details, primed = [], [], []
    for case in (0, 1):
        baseline = read(f"case-{case}-base.json")
        assert len(baseline["bank"]["probes"]) == len(baseline["outputs"]) == 8
        for condition in CONDITIONS:
            edited = read(f"case-{case}-{condition}.json")
            assert edited["bank"] == baseline["bank"]
            counts, decisions = scope_counts(baseline["bank"], baseline["outputs"], edited["after"])
            cases.append({"case": case, "condition": condition, **counts})
            details.extend({"case": case, "condition": condition, **r} for r in decisions)
        base, edited = (
            read(f"case-{case}-primed-base.json"),
            read(f"case-{case}-primed-consensus.json"),
        )
        assert base["probes"] == edited["probes"] and len(base["probes"]) == 3
        assert base["answer_provided_in_input"] and edited["answer_provided_in_input"]
        primed.extend(
            {
                "case": case,
                "index": i,
                "exact_tail_agreement": b == a,
                "base_truncated": b["truncated"],
                "edited_truncated": a["truncated"],
            }
            for i, (b, a) in enumerate(zip(base["tails"], edited["tails"], strict=True))
        )
    groups = {}
    for condition in CONDITIONS:
        rows = [r for r in cases if r["condition"] == condition]
        groups[condition] = {
            key: sum(row[key] for row in rows)
            for key in rows[0]
            if key not in ("case", "condition")
        }
    return {
        "scope": "Two exposed edits; 16 exact-template off-subject probes; no training",
        "interpretation": (
            "Any new target leakage is observed scope failure. No leakage in this small "
            "bank is not proof of specificity. Reference prefixes are benchmark labels; "
            "changed continuations are not uniformly damage. Primed answers are supplied "
            "in the input and never count as edit successes."
        ),
        "groups": groups,
        "cases": cases,
        "primed_tail_checks": primed,
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
