"""Summarize the baseline probe without treating strict mismatches as semantic errors."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(predictions):
    groups = {}
    for row in predictions:
        key = f"{row['task']}/{row['frame']}/{row['order']}"
        group = groups.setdefault(key, {"correct": 0, "n": 0, "unparsed": 0, "truncated": 0})
        group["n"] += 1
        group["correct"] += bool(row["correct"])
        group["unparsed"] += row["parsed"] is None
        group["truncated"] += row["truncated"]
    cases = {}
    for row in predictions:
        if row["task"] != "counterfactual":
            continue
        case = cases.setdefault(row["source_id"], {})
        key = f"{row['frame']}/{row['order']}"
        assert key not in case
        case[key] = row
    matrix = []
    for source, case in cases.items():
        closed = case["closed_book/original"]
        standard = case["context_standard/original"]
        explicit = case["context_explicit/original"]
        world = case["world_with_context/original"]
        item = {
            "source_id": source,
            "truth": closed["truth_letter"],
            "claim": closed["claim_letter"],
            "closed_book_correct": closed["correct"],
            "standard_correct": standard["correct"],
            "explicit_correct": explicit["correct"],
            "world_correct": world["correct"],
            "standard_chooses_truth": standard["parsed"] == closed["truth_letter"],
            "world_chooses_claim": world["parsed"] == closed["claim_letter"],
            "responses": {key: row["text"] for key, row in case.items()},
        }
        for frame in ("context_standard", "context_explicit"):
            rotated = case.get(f"{frame}/rotated")
            if rotated:
                original = case[f"{frame}/original"]
                chosen = rotated["parsed"]
                canonical = "BCDA"["ABCD".index(chosen)] if chosen in list("ABCD") else None
                item[frame + "_rotation_same_content"] = (
                    canonical is not None and canonical == original["parsed"]
                )
                item[frame + "_rotated_correct"] = rotated["correct"]
        matrix.append(item)
    counts = {
        "questions": len(matrix),
        "closed_book_correct": sum(r["closed_book_correct"] for r in matrix),
        "standard_wrong_with_closed_book_correct": sum(
            r["closed_book_correct"] and not r["standard_correct"] for r in matrix
        ),
        "both_reporting_frames_wrong_with_closed_book_correct": sum(
            r["closed_book_correct"] and not r["standard_correct"] and not r["explicit_correct"]
            for r in matrix
        ),
        "explicit_wrong_with_closed_book_correct": sum(
            r["closed_book_correct"] and not r["explicit_correct"] for r in matrix
        ),
        "standard_wrong_explicit_fixed_with_closed_book_correct": sum(
            r["closed_book_correct"] and not r["standard_correct"] and r["explicit_correct"]
            for r in matrix
        ),
        "world_wrong_with_closed_book_correct": sum(
            r["closed_book_correct"] and not r["world_correct"] for r in matrix
        ),
        "world_chooses_fabrication_with_closed_book_correct": sum(
            r["closed_book_correct"] and r["world_chooses_claim"] for r in matrix
        ),
    }
    return {
        "strict_groups": groups,
        "counterfactual_counts": counts,
        "primary_query_count": len(predictions),
        "task_counts": dict(Counter(r["task"] for r in predictions)),
        "interpretation": (
            "Strict generated-answer diagnostics only; "
            "feasibility remains unproven until semantic/source audit."
        ),
    }, matrix


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads((args.results / "completion.json").read_text())
    assert completion["status"] == "all frozen baseline queries complete"
    assert completion["weights_unchanged"] and completion["training_steps"] == 0
    predictions = json.loads((args.results / "predictions.json").read_text())
    assert len(predictions) == completion["queries"] == 300
    summary, matrix = summarize(predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("summary.json", summary),
        ("counterfactual_matrix.json", matrix),
        ("strict_mismatches.json", [r for r in predictions if not r["correct"]]),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
