"""Collect every capped continuation for private review after verification."""

import argparse
import json
from pathlib import Path

from experiments.subject_binding.review import group_outputs


def packet(root, *, partial=False):
    inputs = json.loads((root / "inputs.json").read_text())
    completion = None if partial else json.loads((root / "completion.json").read_text())
    protocol = json.loads((root / "identity.json").read_text())["protocol"]
    available = None
    if partial:
        assert not (root / "completion.json").exists()
        available = {
            (r["case"], r["condition"])
            for r in json.loads((root / "interventions.json").read_text())
        }
    else:
        assert completion["status"] == "complete"
        assert completion["new_fits"] == protocol["new_fits"]
    conditions = ["base", *[c["id"] for c in protocol["conditions"]]]
    records = []
    expected = 0
    for case in inputs["cases"]:
        c = case["case"]
        banks = [
            ("edit", case["edit_probes"]),
            ("utility", inputs["utility"]),
            *((k, case[k]) for k in ("calibration", "evaluation", "exposed_scope")),
        ]
        for condition in conditions:
            if partial and condition != "base" and (c, condition) not in available:
                continue
            expected += sum(len(probes) for _, probes in banks)
            output = json.loads((root / f"case-{c}-{condition}.json").read_text())
            for bank, probes in banks:
                for index, (probe, answer) in enumerate(zip(probes, output[bank], strict=True)):
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
                            "output": answer,
                        }
                    )
    assert len(records) == expected
    grouped = group_outputs(records)
    assert sum(len(g["occurrences"]) for g in grouped) == expected
    return {
        "status": "Private review packet; generated assertions are not facts",
        "partial_grid": partial,
        "limitations": (
            "Unverified benchmark references; 40-token continuations can be incomplete. "
            "Prefix retention and changed text do not establish complete-answer quality. "
            "Review each unique prompt/reference/output context; do not publish raw allegations."
        ),
        "observations": expected,
        "unique_context_outputs": len(grouped),
        "items": grouped,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = packet(args.results, partial=args.partial)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
