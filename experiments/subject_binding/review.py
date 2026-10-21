"""Make a private full-output review packet, deduplicating only equal contexts."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.fine_transfer.scope_analysis import digest_output


def group_outputs(records):
    """Identical continuations for different questions must remain separate items."""
    groups = {}
    for record in records:
        context = [record["prompt"], record["reference_answers"], digest_output(record["output"])]
        key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        if key not in groups:
            groups[key] = {
                "review_id": key,
                "prompt": record["prompt"],
                "reference_answers": record["reference_answers"],
                "output": record["output"],
                "occurrences": [],
            }
        groups[key]["occurrences"].append(
            {k: v for k, v in record.items() if k not in ("prompt", "reference_answers", "output")}
        )
    return list(groups.values())


def packet(root):
    inputs = json.loads((root / "inputs.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["new_fits"] == 6
    records = []
    for case in inputs["cases"]:
        c = case["case"]
        for condition in ("base", "w0", "w1", "w10"):
            row = json.loads((root / f"case-{c}-{condition}.json").read_text())
            for bank, probes in [
                ("edit", case["edit_probes"]),
                ("utility", inputs["utility"]),
                *((k, case[k]) for k in ("calibration", "evaluation", "exposed_scope")),
            ]:
                for index, (probe, output) in enumerate(zip(probes, row[bank], strict=True)):
                    reference = (
                        case["target"]
                        if bank == "edit"
                        else probe.get("gold", probe.get("reference_answers"))
                    )
                    records.append(
                        {
                            "case": c,
                            "condition": condition,
                            "bank": bank,
                            "index": index,
                            "source_index": probe.get("source_index"),
                            "prompt": probe["prompt"],
                            "reference_answers": reference,
                            "output": output,
                        }
                    )
    assert len(records) == 424
    grouped = group_outputs(records)
    assert sum(len(g["occurrences"]) for g in grouped) == len(records)
    return {
        "status": "Private review packet; generated assertions are not facts",
        "limitations": (
            "Benchmark references unverified; equality of prefix scores is not full quality. "
            "Deduplication retains prompt, reference and exact output identity. "
            "Do not publish raw allegations."
        ),
        "observations": len(records),
        "unique_context_outputs": len(grouped),
        "items": grouped,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = packet(args.results)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
