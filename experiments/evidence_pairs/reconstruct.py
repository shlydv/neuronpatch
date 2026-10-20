"""Reconstruct exact candidates from public IDs/offsets and pinned source data.

The specification contains no raw passages. Historical local artifact directories
are unnecessary for reconstruction; their hashes remain provenance metadata.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.evidence_pairs.prepare import SOURCE_SHA256, digest, without_unit, write


def reconstruct(specification, source):
    original = {
        qa["id"]: (article["title"], paragraph, qa)
        for article in source["data"]
        for paragraph in article["paragraphs"]
        for qa in paragraph["qas"]
    }
    datasets = {}
    for name, specifications in specification["candidates"].items():
        rows = []
        for entry in specifications:
            row = copy.deepcopy(entry)
            article, paragraph, qa = original[row["pair_id"]]
            context = paragraph["context"]
            if (
                article != row["article"]
                or hashlib.sha256(context.encode()).hexdigest() != row["original_context_sha256"]
            ):
                raise ValueError("Source context/article mismatch")
            spans = row["removal_spans"]
            if (
                spans[0][0] != 0
                or spans[-1][1] != len(context)
                or any(not 0 <= a < b <= len(context) for a, b in spans)
                or any(left[1] != right[0] for left, right in zip(spans, spans[1:]))
            ):
                raise ValueError("Invalid source removal units")
            answers = list(dict.fromkeys(answer["text"] for answer in qa["answers"]))
            row.update(question=qa["question"], answers=answers, original_context=context)
            for field, unit in (
                ("answer_sentence", "answer_unit"),
                ("control_sentence", "control_unit"),
            ):
                row[field] = context[slice(*spans[row[unit]])].strip()
            for kind, removed in (
                ("answerable", row["control_unit"]),
                ("missing", row["answer_unit"]),
            ):
                row["variants"][kind].update(
                    question=qa["question"],
                    context=without_unit(context, spans, removed),
                    answers=answers if kind == "answerable" else [],
                    answer=answers[0] if kind == "answerable" else "UNKNOWN",
                )
            rows.append(row)
        body = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
        if hashlib.sha256(body.encode()).hexdigest() != specification["manifest"]["sha256"][name]:
            raise ValueError(f"Reconstructed candidate bytes differ: {name}")
        datasets[name] = rows
    return datasets


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.source) != SOURCE_SHA256:
        raise ValueError("Pinned official training source required")
    specification = json.loads(args.spec.read_text())
    if specification["manifest"]["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Specification source differs")
    datasets = reconstruct(specification, json.loads(args.source.read_text()))
    args.output.mkdir(parents=True)
    for name, rows in datasets.items():
        write(args.output / f"{name}_candidates.json", rows)
    write(args.output / "manifest.json", specification["manifest"])
    print(json.dumps({name: len(rows) for name, rows in datasets.items()}, indent=2))


if __name__ == "__main__":
    main()
