"""Collect every complete/capped answer for private full-context review."""

import argparse
import hashlib
import json
from pathlib import Path

from .data import messages
from .run import BANKS


def packet(root):
    def read(name):
        return json.loads((root / name).read_text())

    data = read("inputs.json")
    protocol = read("identity.json")["protocol"]
    completion = read("completion.json")
    assert completion["status"] == "complete" and completion["new_fits"] == protocol["new_fits"]
    conditions = ["base", *[r["id"] for r in protocol["conditions"]]]
    groups = {}
    expected = 0
    for case in data["cases"]:
        c = case["case"]
        for condition in conditions:
            generated = read(f"case-{c}-{condition}.json")
            for bank in BANKS:
                expected += len(case[bank])
                for row, output in zip(case[bank], generated[bank], strict=True):
                    assert output["id"] == row["id"]
                    answer = {k: v for k, v in output.items() if k != "id"}
                    context = [messages(row), row["answers"], answer]
                    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                    if key not in groups:
                        groups[key] = {
                            "review_id": key,
                            "messages": messages(row),
                            "reference_answers": row["answers"],
                            "output": answer,
                            "occurrences": [],
                        }
                    groups[key]["occurrences"].append(
                        {"case": c, "condition": condition, "bank": bank, "id": row["id"]}
                    )
    items = list(groups.values())
    assert sum(len(r["occurrences"]) for r in items) == expected
    return {
        "status": "Private review packet; creating this is not completed qualitative review",
        "observations": expected,
        "unique_context_outputs": len(items),
        "full_output_review_complete": False,
        "limitations": (
            "Counterfactual targets are not real-world facts. Review full answers, "
            "aliases, extra claims and non-stopping outputs. "
            "Do not publish raw personal allegations."
        ),
        "items": items,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = packet(args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
