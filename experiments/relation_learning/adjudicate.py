"""Apply explicit semantic decisions to complete responses, failing on review gaps.

This is bookkeeping, not an automatic semantic judge. Nonexact, complete,
nonempty responses require a signed-off decision tied to their text hash.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.paired_mlp.data import whole_hit


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def adjudicate(case, predictions, decisions):
    """Return paired semantic transitions; never substitute exact-match misses.

    Structural failures cannot be promoted to successes. Reference-matching
    complete responses are accepted automatically unless explicitly reviewed.
    Every other answer needs an explicit decision with a reason. Reject stale,
    duplicate or unused decisions rather than silently changing the denominator.
    """
    judgments = {}
    for entry in decisions:
        key = (entry["model"], entry["id"])
        if key in judgments:
            raise ValueError("Duplicate semantic decision")
        if type(entry["correct"]) is not bool or not entry.get("reason", "").strip():
            raise ValueError("Explicit boolean correctness and review reason required")
        judgments[key] = entry
    if "base" not in predictions:
        raise ValueError("Base predictions required for paired retention")
    records, used, missing = [], set(), []
    for model, banks in predictions.items():
        for bank, inputs in case.items():
            if not isinstance(inputs, list):
                continue
            if [r["id"] for r in banks[bank]] != [r["id"] for r in inputs]:
                raise ValueError("Prediction IDs differ from frozen input order")
            for source, output in zip(inputs, banks[bank], strict=True):
                key = (model, source["id"])
                text_hash = digest(output["text"])
                decision = judgments.get(key)
                if decision is not None:
                    used.add(key)
                    if decision["response_sha256"] != text_hash:
                        raise ValueError("Semantic decision refers to different response text")
                complete = output["ended"] and not output["truncated"]
                valid = complete and not output["unexpected_special"]
                empty = not output["text"].strip()
                exact = valid and not empty and whole_hit(output, source["answers"])
                if not valid or empty:
                    if decision is not None and decision["correct"]:
                        raise ValueError("Incomplete, special-token or empty answer cannot pass")
                    correct = False
                    basis = "structural failure"
                elif decision is not None:
                    correct, basis = decision["correct"], decision["reason"]
                elif exact:
                    correct, basis = True, "complete reference match"
                else:
                    missing.append(
                        {"model": model, "id": source["id"], "response_sha256": text_hash}
                    )
                    continue
                records.append(
                    {
                        "model": model,
                        "bank": bank,
                        "id": source["id"],
                        "family": source.get("family"),
                        "variant": source.get("variant"),
                        "evidence_condition": source.get("condition"),
                        "response_sha256": text_hash,
                        "correct": correct,
                        "exact": exact,
                        "basis": basis,
                        "empty": empty,
                        "incomplete": not complete,
                    }
                )
    if set(judgments) != used:
        raise ValueError("Review contains decisions for absent responses")
    if missing:
        raise ValueError(f"Semantic review incomplete: {json.dumps(missing)}")
    base = {r["id"]: r for r in records if r["model"] == "base"}
    if len(base) != sum(len(rows) for rows in case.values() if isinstance(rows, list)):
        raise ValueError("Frozen input IDs must be globally unique")
    groups = defaultdict(list)
    for record in records:
        groups[record["model"], record["bank"], "all"].append(record)
        if record["variant"] is not None:
            groups[record["model"], record["bank"], record["variant"]].append(record)
    summary = []
    for (model, bank, variant), rows in groups.items():
        summary.append(
            {
                "model": model,
                "bank": bank,
                "variant": variant,
                "count": len(rows),
                "correct": sum(r["correct"] for r in rows),
                "base_correct": sum(base[r["id"]]["correct"] for r in rows),
                "corrections": sum(r["correct"] and not base[r["id"]]["correct"] for r in rows),
                "regressions": sum(not r["correct"] and base[r["id"]]["correct"] for r in rows),
                "empty": sum(r["empty"] for r in rows),
                "incomplete": sum(r["incomplete"] for r in rows),
            }
        )
    return {"records": records, "summary": summary}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_inputs = (args.results / "inputs.json").read_bytes()
    raw_review = args.review.read_bytes()
    review = json.loads(raw_review)
    if review["inputs_sha256"] != hashlib.sha256(raw_inputs).hexdigest():
        raise ValueError("Review belongs to a different frozen input set")
    fits = json.loads((args.results / "fits.json").read_text())
    labels = ["base", *[fit["condition"]["id"] for fit in fits]]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate model labels")
    predictions = {
        label: json.loads((args.results / f"case-0-{label}.json").read_text()) for label in labels
    }
    result = adjudicate(json.loads(raw_inputs)["cases"][0], predictions, review["decisions"])
    result.update(
        inputs_sha256=review["inputs_sha256"],
        review_sha256=hashlib.sha256(raw_review).hexdigest(),
        scope="Explicit semantic review plus complete reference matches; not an independent judge",
    )
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"reviewed_outputs": len(result["records"])}))


if __name__ == "__main__":
    main()
