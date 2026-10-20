"""Find training answers omitted by exact-match preservation for manual review.

Containing a reference span is not proof of correctness: negation, contradictory
alternatives and copied passages can all contain it. This creates a review queue,
never new training labels or an alternative accuracy metric.
"""

import hashlib
import json
from pathlib import Path

from data import normalize

ROOT = Path(__file__).parent


def contains_answer(output, answers):
    tokens = normalize(output).split()
    for answer in answers:
        target = normalize(answer).split()
        if target and any(
            tokens[i : i + len(target)] == target for i in range(len(tokens) - len(target) + 1)
        ):
            return True
    return False


def main():
    baseline_path = ROOT / "scaled_baseline/train.json"
    training_path = ROOT / "scaled/train_expanded.json"
    baseline = json.loads(baseline_path.read_text())
    training = json.loads(training_path.read_text())
    assert len(baseline) == len(training) == 2048
    assert len({r["id"] for r in baseline}) == len(baseline)
    for original, prediction in zip(training, baseline):
        for key in ("id", "kind", "context", "question", "answers"):
            assert original[key] == prediction[key], (original["id"], key)
    candidates = [
        r
        for r in baseline
        if r["kind"] == "answerable"
        and not r["abstained"]
        and r["em"] == 0
        and contains_answer(r["output"], r["answers"])
    ]
    report = {
        "purpose": "Training-only review queue; no automatic correctness judgments",
        "changes_anchor_policy": False,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (baseline_path, training_path)
        },
        "prefixes": {},
        "candidates": candidates,
    }
    for size in (128, 512, 2048):
        prefix = baseline[:size]
        ids = {r["id"] for r in prefix}
        queued = [r for r in candidates if r["id"] in ids]
        entry = {
            "supported": sum(r["kind"] == "answerable" for r in prefix),
            "exact_anchors": sum(r["kind"] == "answerable" and r["em"] == 1 for r in prefix),
            "unreviewed_span_containing_nonexact": len(queued),
            "checkpoint_outcomes": {},
        }
        for path in sorted((ROOT / f"scale_{size}").glob("*-train.json")):
            adapted = {r["id"]: r for r in json.loads(path.read_text())}
            assert set(adapted) == ids
            for r in prefix:
                assert all(adapted[r["id"]][k] == r[k] for k in ("context", "question", "answers"))
            entry["checkpoint_outcomes"][path.name] = {
                "candidate_refusals": sum(adapted[r["id"]]["abstained"] for r in queued),
                "candidate_exact_matches": sum(adapted[r["id"]]["em"] == 1 for r in queued),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        report["prefixes"][str(size)] = entry
    destination = ROOT / "anchor_coverage"
    destination.mkdir(exist_ok=True)
    (destination / "review_queue.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["prefixes"], indent=2))


if __name__ == "__main__":
    main()
