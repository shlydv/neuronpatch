"""Freeze a development-only GSM8K slice before any model outputs are inspected."""

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

SOURCE_SHA = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
NUMBER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")
SYSTEM = (
    "Solve the math problem. Show your reasoning, then end with a line containing "
    "#### followed by only the final numeric answer."
)


def final_number(text):
    """Strict marked final answer; unparseable output requires separate review.

    Never take the last arbitrary number in reasoning as a correct answer.
    Multiple markers, trailing text and nonstandard formatting remain unparsed,
    not automatically mathematical failures.
    """
    if text.count("####") != 1:
        return None
    prefix, value = text.split("####")
    if prefix and not prefix.endswith("\n"):
        return None
    value = value.strip()
    return str(Decimal(value.replace(",", ""))) if NUMBER.fullmatch(value) else None


def partition(rows, sizes, *, salt="neuronpatch-math-development-v1"):
    keys = [" ".join(row["question"].lower().split()) for row in rows]
    counts = Counter(keys)
    candidates = []
    for row, key in zip(rows, keys, strict=True):
        if counts[key] != 1:
            continue  # Drop whole duplicate-question groups, including conflicting labels.
        response = re.sub(r"<<.*?>>", "", row["answer"])
        number = final_number(response)
        if number is None:
            raise ValueError("Unparseable source answer")
        identity = hashlib.sha256(key.encode()).hexdigest()
        order = hashlib.sha256((salt + ":" + identity).encode()).hexdigest()
        candidates.append(
            (
                order,
                {
                    "id": identity,
                    "question": row["question"],
                    "response": response,
                    "final_number": number,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": row["question"]},
                    ],
                },
            )
        )
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes.values()):
        raise ValueError("Positive split sizes required")
    if sum(sizes.values()) > len(candidates):
        raise ValueError("Insufficient unique source questions")
    candidates.sort(key=lambda item: item[0])
    splits, start = {}, 0
    for name, size in sizes.items():
        splits[name] = [row for _, row in candidates[start : start + size]]
        start += size
    return {
        "source_rows": len(rows),
        "unique_question_rows": len(candidates),
        "excluded_duplicate_rows": len(rows) - len(candidates),
        "split_salt": salt,
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Source differs from the pinned official training file")
    result = partition(
        [json.loads(line) for line in raw.splitlines()],
        {"train": 128, "calibration": 32, "development": 128},
    )
    result["source_sha256"] = SOURCE_SHA
    result["official_test_used"] = False
    # All these splits are development; none is advertised as final confirmation.
    args.output.mkdir(parents=True, exist_ok=False)
    content = (json.dumps(result, indent=2) + "\n").encode()
    (args.output / "inputs.json").write_bytes(content)
    (args.output / "frozen.json").write_text(
        json.dumps(
            {
                "inputs_sha256": hashlib.sha256(content).hexdigest(),
                "source_sha256": SOURCE_SHA,
                "stage": "baseline qualification only; no adaptation run authorized by this file",
                "qualification": {
                    "development_examples": 128,
                    "minimum_completed_mathematically_wrong": 16,
                    "unparsed_and_incomplete": (
                        "review separately; do not count as mathematical errors"
                    ),
                    "failure_action": (
                        "no training on this slice; record the ceiling or execution limit"
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
