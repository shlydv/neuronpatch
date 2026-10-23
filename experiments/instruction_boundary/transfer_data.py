"""Fresh documents and clean-teacher targets for exploratory boundary adaptation.

This explicitly follows a failed baseline gate; it does not rewrite that gate.
No reference answer is used as a training target. Dataset references are retained
only for later semantic evaluation and may themselves require correction.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .prepare import build, context_key


def fresh_documents(rows, excluded, sizes):
    excluded = set(excluded)
    remaining = [row for row in rows if context_key(row["context"]) not in excluded]
    result = build(remaining, sizes)
    if excluded & {row["document_id"] for row in result["membership"]}:
        raise ValueError("Previously allocated document leaked into fresh experiment")
    result["excluded_documents"] = sorted(excluded)
    return result


def teacher_supervision(cases, predictions):
    """Use completed clean outputs for redirect targets and clean preservation.

    Input must be one training/calibration bank, never the confirmation bank.
    Predictions contain exactly one clean response per document. Every completed
    nonempty response is retained regardless of reference agreement. Incomplete
    teachers are excluded from both task and preservation banks and recorded.
    Direct and quoted controls target their legitimate marker. Claimed-authority
    cases are held out from both optimization and rate selection.
    """
    clean = {row["id"]: row for row in cases if row["kind"] == "clean"}
    if len({r["id"] for r in cases}) != len(cases) or not clean:
        raise ValueError("Unique cases including clean teachers required")
    outputs = {row["id"]: row for row in predictions}
    if len(outputs) != len(predictions) or set(outputs) != set(clean):
        raise ValueError("Exactly one prediction for every clean teacher required")
    targets, excluded = {}, []
    for key, row in clean.items():
        output = outputs[key]
        if type(output["ended"]) is not bool or not isinstance(output["text"], str):
            raise ValueError("Explicit termination and text required")
        if output["ended"] and output["text"].strip():
            targets[row["document_id"]] = output["text"]
        else:
            excluded.append(row["document_id"])
    task, preserve = [], []
    for row in cases:
        if row["document_id"] not in targets:
            continue
        kind, family = row["kind"], row["family"]
        if kind == "clean" or (kind == "injected" and family == "redirect"):
            target = targets[row["document_id"]]
        elif kind in ("direct", "quoted") and family == "redirect":
            target = row["marker"]
        else:
            continue
        item = {
            "id": row["id"],
            "document_id": row["document_id"],
            "messages": row["messages"],
            "target": target,
            "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        }
        (task if kind == "injected" else preserve).append(item)
    if len(task) != len(targets) or len(preserve) != 3 * len(targets):
        raise ValueError("Incomplete redirect task/control pairs")
    return {"task": task, "preserve": preserve, "excluded_documents": sorted(excluded)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(Path("experiments/instruction_boundary/frozen_v1.json").read_text())
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != original["source_sha256"]:
        raise ValueError("Pinned source mismatch")
    result = fresh_documents(
        [json.loads(line) for line in raw.splitlines()],
        [row["document_id"] for row in original["membership"]],
        {"train": 32, "calibration": 16, "confirmation": 48},
    )
    content = (json.dumps(result, indent=2) + "\n").encode()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.json").write_bytes(content)
    manifest = {
        "stage": "fresh-data freeze only; optimizer and selection protocol not yet frozen",
        "source_sha256": original["source_sha256"],
        "inputs_sha256": hashlib.sha256(content).hexdigest(),
        "excluded_documents": result["excluded_documents"],
        "membership": result["membership"],
        "training_family": "redirect",
        "held_out_instruction_family": "claimed_authority",
        "targets": "completed clean-model responses; no factual-correctness filtering",
        "changes_from_baseline": "new documents; max generation256 instead of128 for all arms",
        "interpretation": "Exploratory follow-up, not a retroactive qualification pass",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
