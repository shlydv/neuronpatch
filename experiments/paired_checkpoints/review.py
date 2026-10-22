"""Collect every early/full answer with its full context;collection is not review."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.data import messages


def packet(original, results):
    spec = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    assert completion["status"] == "complete"
    cases = {c["case"]: c for c in json.loads((original / "inputs.json").read_text())["cases"]}
    fits = json.loads((results / "fits.json").read_text())
    groups = {}
    total = 0
    for fit in fits:
        c, name = fit["case"], fit["condition"]["id"]
        for state in fit["states"]:
            generated = json.loads((results / f"case-{c}-{name}-{state}.json").read_text())
            for bank in spec["generation_banks"]:
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
                            "role": fit["role"],
                            "state": state,
                            "bank": bank,
                            "id": row["id"],
                        }
                    )
                    total += 1
    assert total == completion["observations"]
    return {
        "status": "Private review packet;not a completed qualitative review",
        "observations": total,
        "unique_context_outputs": len(groups),
        "full_output_review_complete": False,
        "limits": (
            "Counterfactual targets are not real-world facts. Review valid aliases,stopping,"
            "additional claims and genuine unrelated errors."
        ),
        "items": list(groups.values()),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("original", "results", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = packet(args.original, args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
