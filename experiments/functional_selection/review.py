"""Collect full contexts and outputs; collection is not a qualitative review."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.data import messages


def packet(parent, results):
    protocol = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    assert completion["status"] == "complete"
    data = json.loads((parent / "inputs.json").read_text())["cases"]
    records = json.loads((results / "records.json").read_text())
    groups = {}
    observations = 0
    for case in data:
        c = case["case"]
        sources = [
            (
                {"condition": "base", "case": c, "scale": 0},
                json.loads((parent / f"case-{c}-base.json").read_text()),
            )
        ]
        sources.extend(
            (r, json.loads((results / f"{r['prefix']}-generations.json").read_text()))
            for r in records
            if r["case"] == c and r["generated"]
        )
        for source, outputs in sources:
            for bank in protocol["generation_banks"]:
                for row, answer in zip(case[bank], outputs[bank], strict=True):
                    assert row["id"] == answer["id"]
                    output = {k: v for k, v in answer.items() if k != "id"}
                    context = [messages(row), row["answers"], output]
                    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                    item = groups.setdefault(
                        key,
                        {
                            "review_id": key,
                            "messages": messages(row),
                            "reference_answers": row["answers"],
                            "output": output,
                            "occurrences": [],
                        },
                    )
                    item["occurrences"].append(
                        {
                            "case": c,
                            "condition": source["condition"],
                            "scale": source["scale"],
                            "bank": bank,
                            "id": row["id"],
                        }
                    )
                    observations += 1
    assert observations == completion["observations"]
    return {
        "observations": observations,
        "unique_context_outputs": len(groups),
        "full_output_review_complete": False,
        "items": list(groups.values()),
        "scope": (
            "Private review packet. Counterfactual targets are not real facts; "
            "review complete answers,aliases,stopping and collateral errors."
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = packet(args.parent, args.results)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
