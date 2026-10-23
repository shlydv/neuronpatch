"""Create a deterministic review packet before any model predictions.

Read an independently downloaded, pinned RGB source. Do not copy upstream
evaluation code, truncate documents, flatten conjunctive answers, or treat a
negative-document annotation as a verified absence of evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def answer_groups(answer):
    """Outer items are jointly required; inner strings are alternatives."""
    if not isinstance(answer, list) or not answer:
        raise ValueError("Expected a nonempty outer answer list")
    groups = [[part] if isinstance(part, str) else part for part in answer]
    if any(
        not isinstance(group, list)
        or not group
        or any(not isinstance(text, str) or not text.strip() for text in group)
        for group in groups
    ):
        raise ValueError("Invalid answer group")
    return groups


def prepare(source, manifest, count):
    raw = source.read_bytes()
    expected = manifest["files"]["data/en_refine.json"]["sha256"]
    if sha256(raw) != expected:
        raise ValueError("Source checksum mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids) or not 1 <= count <= len(rows):
        raise ValueError("Duplicate source IDs or invalid sample size")
    for row in rows:
        answer_groups(row["answer"])
        for field, minimum in (("positive", 1), ("negative", 3)):
            if len(row[field]) < minimum or any(
                not isinstance(text, str) or not text.strip() for text in row[field]
            ):
                raise ValueError(f"Invalid documents for {row['id']}")
    # No model failures, answer types, topic choices or source-label quality are
    # used to rank examples. Rejections must be explicit in a later review.
    ordered = sorted(
        rows,
        key=lambda row: sha256(f"retrieval-evidence-v1:43:{row['id']}".encode()),
    )
    items = []
    for row in ordered[:count]:
        slot = int(sha256(str(row["id"]).encode())[:8], 16) % 3
        missing = row["negative"][:3]
        supported = list(missing)
        supported[slot] = row["positive"][0]
        items.append(
            {
                "source_id": row["id"],
                "question": row["query"],
                "answer_groups": answer_groups(row["answer"]),
                "replacement_slot": slot,
                "supported_documents": supported,
                "missing_documents": missing,
                "document_sha256": {
                    "supported": [sha256(x.encode()) for x in supported],
                    "missing": [sha256(x.encode()) for x in missing],
                },
                "review_status": "unreviewed",
            }
        )
    return {
        "source_revision": manifest["revision"],
        "source_sha256": expected,
        "source_rows": len(rows),
        "sample_count": count,
        "license": "RGB source: CC BY-NC-SA 4.0; research data, not library code",
        "status": "Candidate review packet only; not a frozen GPU protocol",
        "selection": "SHA256 ordering of retrieval-evidence-v1:43:source_id",
        "limits": (
            "Documents are unchanged. Length is not matched. Annotation alone does not "
            "prove supported or missing evidence. Review all candidates, record exclusions, "
            "and check token budgets before inference. No claims of unseen pretraining data."
        ),
        "items": items,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    packet = prepare(args.source, json.loads(args.manifest.read_text()), args.count)
    with args.output.open("x") as stream:
        stream.write(json.dumps(packet, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in packet.items() if k != "items"}))


if __name__ == "__main__":
    main()
