"""Freeze reviewed clean teachers for a future matched distillation comparison.

No fitting, GPU submission, or answer generation. This preserves raw trajectories
and checks their recorded origin; runtime must still verify loaded model identity.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .teacher_review import approved_teachers


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(inventory_path, review_path, parent_protocol_path, teacher_dir, protocol_path):
    paths = [inventory_path, review_path, parent_protocol_path, protocol_path]
    inventory, review, parent, protocol = [json.loads(p.read_text()) for p in paths]
    for field in ("model", "revision"):
        if parent[field] != protocol[field]:
            raise ValueError(f"teacher parent {field} mismatch")
    origins = [
        entry
        for entry in parent["reused_parents"]
        if all(f"teachers-{split}.json" in entry["files_sha256"] for split in inventory)
    ]
    if len(origins) != 1:
        raise ValueError("require one unambiguous recorded teacher origin")
    origin = origins[0]
    for split, records in inventory.items():
        source = teacher_dir / f"teachers-{split}.json"
        if sha256(source) != origin["files_sha256"][source.name]:
            raise ValueError("teacher source hash mismatch")
        paths.append(source)
        originals = json.loads(source.read_text())
        by_id = {record["id"]: record for record in originals}
        if len(by_id) != len(originals):
            raise ValueError("duplicate original teacher IDs")
        for record in records:
            if by_id.get(record["teacher"]["id"]) != record["teacher"]:
                raise ValueError("inventory trajectory differs from original")
    approved, receipt = approved_teachers(inventory, review)
    if any(not records for records in approved.values()):
        raise ValueError("no approved teachers in a split")
    return {
        "status": "reviewed teacher preparation only; not a submitted experiment",
        "model": protocol["model"],
        "revision": protocol["revision"],
        "recorded_base_fingerprint": origin["base_fingerprint"],
        "origin_kernel": origin["kernel"],
        "origin_version": origin["kernel_version"],
        "sources": {str(path): sha256(path) for path in paths},
        "teachers": approved,
        "review_receipt": receipt,
        "runtime_requirements": [
            "Verify loaded base fingerprint and pinned tokenizer.",
            "Recompute each clean prompt's token IDs and compare with teacher prompt_ids.",
            "Verify response decode and terminal EOS against pinned tokenizer.",
            "Use identical approved banks for each objective comparator.",
            "Report excluded documents; do not filter evaluation using model correctness.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for field in ("inventory", "review", "parent-protocol", "teacher-dir", "protocol", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.inventory, args.review, args.parent_protocol, args.teacher_dir, args.protocol
    )
    # Validate everything before creating an immutable-by-convention new artifact.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({split: len(rows) for split, rows in result["teachers"].items()}))


if __name__ == "__main__":
    main()
