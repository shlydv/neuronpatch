"""Reproduce secondary content counts while preserving the frozen strict results."""

import argparse
import hashlib
import json
from pathlib import Path

from .content import exact_option


def read(path):
    return json.loads(Path(path).read_text())


def completed_predictions(primary, extensions):
    """Use only complete, prefix-verified evaluation continuations in sequence.

    The primary rows are copied, leaving frozen strict scores untouched. Every
    truncated answer must occur in the next extension, without additions, and
    the new tokens must reproduce the full preceding generated sequence.
    """
    rows = [dict(row, response_source="primary") for row in primary]
    for stage, extension in enumerate(extensions, 1):
        expected = [row["id"] for row in rows if row["truncated"]]
        actual = [row["id"] for row in extension]
        if actual != expected or len(set(actual)) != len(actual):
            raise ValueError("Extension must include every truncated answer exactly once")
        lookup = {row["id"]: row for row in extension}
        for row in rows:
            if row["id"] not in lookup:
                continue
            new = lookup[row["id"]]
            prefix = row["generated_token_ids"]
            if new["generated_token_ids"][: len(prefix)] != prefix:
                raise ValueError("Extension changed the original generated prefix")
            row.update(
                text=new["text"],
                generated_token_ids=new["generated_token_ids"],
                truncated=new["truncated"],
                response_source=f"extension{stage}",
            )
    return rows


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(__doc__)
    here = Path(__file__).parent
    parser.add_argument("--source", type=Path, required=True, help="Pinned counterfactual parquet")
    parser.add_argument("--results", type=Path, default=here / "colab_v1")
    parser.add_argument("--reviews", type=Path, default=here / "colab_v1/content_reviews.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    manifest = read(here / "data_manifest_v1.json")
    expected_hash = next(
        r["sha256"]
        for r in manifest["source_files"]
        if r["dataset"] == "Salesforce/FaithEval-counterfactual-v1.0"
        and r["file"].endswith(".parquet")
    )
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Counterfactual source hash differs from frozen manifest")
    primary = completed_predictions(
        read(root / "primary/predictions.json"),
        [read(root / f"extension{n}/predictions.json") for n in (1, 2)],
    )
    roles = read(root / "system_role/predictions.json")
    reviews = read(args.reviews)
    source = {r["id"]: r for r in pq.read_table(args.source).to_pylist()}
    rows = []
    for phase, predictions in [("original", primary), ("system_role", roles)]:
        for row in predictions:
            if row["task"] != "counterfactual":
                continue
            order = [0, 1, 2, 3] if row["order"] == "original" else [1, 2, 3, 0]
            options = source[row["source_id"]]["choices"]["text"]
            review = reviews[phase].get(row["id"])
            choice = (
                review["choice"]
                if review is not None
                else exact_option(row["text"], [options[i] for i in order])
            )
            if review is None and choice is None:
                raise ValueError("Unreviewed content response: " + row["id"])
            rows.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "source_id",
                        "frame",
                        "order",
                        "answer",
                        "claim_letter",
                        "truth_letter",
                    )
                }
                | {
                    "content_choice": choice,
                    "content_correct": choice == row["answer"] if choice else None,
                    "response_source": row.get("response_source", "system_role"),
                    "response_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "truncated": row["truncated"],
                    "scoring": "explicit assistant review"
                    if review
                    else "exact option-content rule",
                }
            )
    groups = {}
    for row in rows:
        key = row["frame"] + "/" + row["order"]
        counts = groups.setdefault(key, {"n": 0, "correct": 0, "wrong": 0, "unresolved": 0})
        counts["n"] += 1
        counts[
            "unresolved"
            if row["content_choice"] is None
            else "correct"
            if row["content_correct"]
            else "wrong"
        ] += 1
    lookup = {(r["source_id"], r["frame"], r["order"]): r for r in rows}
    baseline = [r for r in rows if r["frame"] == "closed_book" and r["content_correct"]]
    retention = {}
    changed = []
    for frame in ("world_with_context", "world_with_context_system"):
        states = [lookup[(r["source_id"], frame, "original")] for r in baseline]
        retention[frame] = {
            "baseline_correct": len(baseline),
            "retained": sum(r["content_correct"] is True for r in states),
            "changed_to_wrong": sum(r["content_correct"] is False for r in states),
            "unresolved": sum(r["content_correct"] is None for r in states),
        }
        changed += [
            {**r, "comparison": "closed-book reference correct"}
            for r in states
            if r["content_correct"] is False
        ]
    summary = {
        "status": "Post-run content diagnostic; not training or confirmation",
        "groups": groups,
        "retention_of_closed_book_correct": retention,
        "primary_strict_results_unchanged": True,
        "one_original_science_response_unresolved_at_1024": True,
        "no_context_router_baseline": (
            "Closed-book template is already an evaluated "
            "no-training route for the explicit factual mode."
        ),
        "caution": (
            "Counterfactual target-label matching is not a fully validated "
            "semantic faithfulness score; source-scope problems are documented in reviews."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("summary.json", summary),
        ("content_decisions.json", rows),
        ("science_interference_cases.json", changed),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
