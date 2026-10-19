"""Compare all completed fits on identical validation IDs, without choosing a winner."""

import json
from pathlib import Path

from data import metrics

ROOT = Path(__file__).parent


def main():
    base = {r["id"]: r for r in json.loads((ROOT / "scaled_baseline/val.json").read_text())}
    old_ids = {r["id"] for r in json.loads((ROOT / "baseline/val.json").read_text())}
    result = {}
    for size in (128, 512, 2048):
        directory = ROOT / f"scale_{size}"
        if not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        for name, run in summary["runs"].items():
            rows = json.loads((directory / f"{name}-val.json").read_text())
            if {r["id"] for r in rows} != set(base):
                raise ValueError("Validation populations differ")
            known_correct = [
                r for r in rows if r["kind"] == "answerable" and base[r["id"]]["em"] == 1
            ]
            result[f"{size}/{name}"] = {
                "training_examples": size,
                "lr": run["lr"],
                "optimizer_steps": run["optimizer_steps_total"],
                "anchor_count": run.get("anchor_count", summary.get("anchor_count")),
                "validation": metrics(rows),
                "new_only": metrics([r for r in rows if r["id"] not in old_ids]),
                "base_exact_correct_supported": len(known_correct),
                "retained_exact_correct_supported": sum(r["em"] for r in known_correct),
                "new_refusals_on_base_exact_correct": sum(r["abstained"] for r in known_correct),
                "training_seconds": run["training_seconds"],
            }
    (ROOT / "scaled/comparison.json").write_text(json.dumps(result, indent=2))
    for name, r in result.items():
        print(
            name,
            r["validation"],
            "retained",
            r["retained_exact_correct_supported"],
            "new refusals on known-correct",
            r["new_refusals_on_base_exact_correct"],
        )


if __name__ == "__main__":
    main()
