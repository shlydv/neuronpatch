"""Summarize common refusal shifts separately from within-pair changes."""

import json
from pathlib import Path

ROOT = Path(__file__).parent / "diagnostic"


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    result = {}
    for name, splits in summary["runs"].items():
        result[name] = {}
        for split, measured in splits.items():
            base = summary["runs"]["base"][split]
            rows = json.loads((ROOT / f"{name}-{split}.json").read_text())
            pairs = list(zip(rows[::2], rows[1::2]))
            result[name][split] = {
                "answerable_margin_change": measured["mean_margin"]["answerable"]
                - base["mean_margin"]["answerable"],
                "missing_margin_change": measured["mean_margin"]["missing"]
                - base["mean_margin"]["missing"],
                "pair_separation_change": measured["mean_pair_separation"]
                - base["mean_pair_separation"],
                "both_exact_correct_pairs": sum(a["em"] == b["em"] == 1 for a, b in pairs),
                "both_refused_pairs": sum(a["abstained"] and b["abstained"] for a, b in pairs),
                "pairs": len(pairs),
            }
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2))
    for name, splits in result.items():
        print(
            name,
            {
                s: {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                for s, r in splits.items()
            },
        )


if __name__ == "__main__":
    main()
