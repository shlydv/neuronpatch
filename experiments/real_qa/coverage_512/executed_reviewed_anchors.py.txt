"""Validate explicit training-only review additions; never infer correctness."""

import hashlib
import json


def evidence_digest(row):
    evidence = {k: row[k] for k in ("id", "context", "question", "answers", "output")}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


def reviewed_additions(training_rows, baseline_rows, manifest):
    """Return reviewed nonexact IDs after checking scope and reviewed evidence.

    The caller unions this set with exact anchors. A review is an explicit human
    or assistant judgment, not a promise of semantic correctness. The digest binds
    that judgment to the actual passage, question, aliases and baseline output.
    """
    training = {r["id"]: r for r in training_rows}
    baseline = {r["id"]: r for r in baseline_rows}
    reviews = {r["id"]: r for r in manifest["reviews"]}
    eligible = manifest["eligible_additional_ids"]
    if len(training) != len(training_rows) or len(baseline) != len(baseline_rows):
        raise ValueError("duplicate training or baseline IDs")
    if len(reviews) != len(manifest["reviews"]) or len(set(eligible)) != len(eligible):
        raise ValueError("duplicate reviewed or eligible IDs")
    accepted = {i for i, r in reviews.items() if r["category"] == "valid_answer_extra_wording"}
    if accepted != set(eligible):
        raise ValueError("eligible IDs disagree with review decisions")
    # Validate every review, including rejected cases, to keep the review set scoped.
    for identifier, review in reviews.items():
        if identifier not in training or identifier not in baseline:
            raise ValueError("reviewed example outside training prefix or absent from baseline")
        row, prediction = training[identifier], baseline[identifier]
        if any(row[k] != prediction[k] for k in ("kind", "context", "question", "answers")):
            raise ValueError("baseline/training evidence mismatch")
        if review.get("evidence_sha256") != evidence_digest(prediction):
            raise ValueError("review evidence changed or digest missing")
        if identifier in accepted and (
            row["kind"] != "answerable"
            or not row["answers"]
            or prediction["abstained"]
            or prediction["em"] != 0
        ):
            raise ValueError("addition must be a supported nonexact nonrefusal")
    return accepted
