"""Describe paired development changes without revising the automatic gold labels."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    baseline = {r["id"]: r for r in json.loads((ROOT / "baseline/val.json").read_text())}
    review = json.loads((ROOT / "validation_review.json").read_text())["categories"]
    categories = {identifier: category for category, ids in review.items() for identifier in ids}
    summary = {}
    for path in sorted((ROOT / "adaptation").glob("*-val.json")):
        rows = json.loads(path.read_text())
        if {r["id"] for r in rows} != set(baseline):
            raise ValueError("Predictions do not match baseline IDs")
        changes = []
        for row in rows:
            old = baseline[row["id"]]
            changes.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "question": row["question"],
                    "answers": row["answers"],
                    "before": old["output"],
                    "after": row["output"],
                    "em_delta": row["em"] - old["em"],
                    "f1_delta": row["f1"] - old["f1"],
                    "new_refusal": row["abstained"] and not old["abstained"],
                    "baseline_review": categories.get(row["id"], "exact_correct_or_impossible"),
                }
            )
        summary[path.stem] = {
            "em_gains": sum(r["em_delta"] > 0 for r in changes),
            "em_losses": sum(r["em_delta"] < 0 for r in changes),
            "new_supported_refusals": sum(
                r["new_refusal"] and r["kind"] == "answerable" for r in changes
            ),
            "wording_only_baseline_em_gains": sum(
                r["em_delta"] > 0 and r["baseline_review"] == "wording_only" for r in changes
            ),
            "changes": changes,
        }
    (ROOT / "adaptation/paired_changes.json").write_text(json.dumps(summary, indent=2))
    for name, result in summary.items():
        print(name, {k: v for k, v in result.items() if k != "changes"})


if __name__ == "__main__":
    main()
