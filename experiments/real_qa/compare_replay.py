"""Compare matched replay fits against their original in-batch-preservation fits."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    previous = json.loads((ROOT / "diagnostic/summary.json").read_text())["runs"]
    current = json.loads((ROOT / "replay/summary.json").read_text())["runs"]
    result = {}
    for name, run in current.items():
        old = previous[name]
        result[name] = {
            "anchor_em_before": old["train"]["anchor_em"],
            "anchor_em_after": run["anchor_em"],
            "anchor_refusals_before": old["train"]["anchor_refusals"],
            "anchor_refusals_after": run["anchor_refusals"],
            "validation_before": old["val"]["metrics"],
            "validation_after": run["validation"],
            "train_before": old["train"]["metrics"],
            "train_after": run["train"],
        }
        before = {
            r["id"]: r for r in json.loads((ROOT / "diagnostic" / f"{name}-train.json").read_text())
        }
        after = json.loads((ROOT / "replay" / f"{name}-train.json").read_text())
        result[name]["anchor_changes"] = [
            {
                "id": r["id"],
                "question": r["question"],
                "answers": r["answers"],
                "before": before[r["id"]]["output"],
                "after": r["output"],
            }
            for r in after
            if before[r["id"]]["anchor"] and r["output"] != before[r["id"]]["output"]
        ]
    (ROOT / "replay/comparison.json").write_text(json.dumps(result, indent=2))
    for name, row in result.items():
        print(
            name,
            {k: v for k, v in row.items() if k.startswith("anchor_") and k != "anchor_changes"},
        )


if __name__ == "__main__":
    main()
