"""Collect all complete outputs for human/assistant review; collection is not review."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.data import messages
from experiments.paired_mlp.run import BANKS


def packet(parent, results):
    data = json.loads((parent / "inputs.json").read_text())
    protocol = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    records = json.loads((results / "records.json").read_text())
    assert completion["status"] == "complete" and len(records) == protocol["new_fits"]
    observations, groups = 0, {}
    for case in data["cases"]:
        c = case["case"]
        sources = [("base", "base", f"case-{c}-base")]
        sources += [
            (r["method"], r["arm"], f"case-{c}-{r['method']}-{r['arm']}")
            for r in records
            if r["case"] == c
        ]
        for method, arm, stem in sources:
            outputs = json.loads((results / f"{stem}-answers.json").read_text())
            for bank in BANKS:
                for row, output in zip(case[bank], outputs[bank], strict=True):
                    assert row["id"] == output["id"]
                    answer = {k: v for k, v in output.items() if k != "id"}
                    context = [messages(row), row["answers"], answer]
                    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                    item = groups.setdefault(
                        key,
                        {
                            "review_id": key,
                            "messages": context[0],
                            "reference_answers": context[1],
                            "output": answer,
                            "occurrences": [],
                        },
                    )
                    item["occurrences"].append(
                        {"case": c, "method": method, "arm": arm, "bank": bank, "id": row["id"]}
                    )
                    observations += 1
    assert observations == completion["observations"]
    return {
        "observations": observations,
        "unique_context_outputs": len(groups),
        "full_output_review_complete": False,
        "items": list(groups.values()),
        "scope": (
            "Private collection, not a completed review. Inspect complete answers, valid aliases, "
            "stopping and collateral errors. Counterfactual targets are not real-world facts."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    value = packet(args.parent, args.results)
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "items"}))


if __name__ == "__main__":
    main()
