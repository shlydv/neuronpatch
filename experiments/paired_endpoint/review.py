"""Collect complete live/merged response contexts for a private qualitative review."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.data import messages


def packet(parent, results):
    protocol = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["new_fits"] == 0
    cases = {c["case"]: c for c in json.loads((parent / "inputs.json").read_text())["cases"]}
    groups = {}
    total = 0
    for request in protocol["endpoints"]:
        c, name = request["case"], request["condition"]["id"]
        for state in ("live", "merged"):
            generated = json.loads((results / f"case-{c}-{name}-{state}.json").read_text())
            for bank in protocol["generation_banks"]:
                for row, output in zip(cases[c][bank], generated[bank], strict=True):
                    assert row["id"] == output["id"]
                    answer = {k: v for k, v in output.items() if k != "id"}
                    context = [messages(row), row["answers"], answer]
                    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                    groups.setdefault(
                        key,
                        {
                            "review_id": key,
                            "messages": messages(row),
                            "reference_answers": row["answers"],
                            "output": answer,
                            "occurrences": [],
                        },
                    )["occurrences"].append(
                        {
                            "case": c,
                            "condition": name,
                            "state": state,
                            "bank": bank,
                            "id": row["id"],
                        }
                    )
                    total += 1
    assert total == protocol["expected_generations"] == completion["observations"]
    return {
        "status": "Private packet; collection is not completed qualitative review",
        "observations": total,
        "unique_context_outputs": len(groups),
        "full_output_review_complete": False,
        "limits": (
            "Counterfactual targets are not factual claims; review aliases and entire responses."
        ),
        "items": list(groups.values()),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = packet(args.parent, args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
