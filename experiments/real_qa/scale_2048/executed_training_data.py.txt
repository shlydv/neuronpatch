"""Training-only preservation anchors for nested data-size comparisons."""


def exact_correct_anchors(training_rows, baseline_rows):
    baseline = {r["id"]: r for r in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError("duplicate baseline IDs")
    if len({r["id"] for r in training_rows}) != len(training_rows):
        raise ValueError("duplicate training IDs")
    anchors = set()
    for row in training_rows:
        if row["id"] not in baseline:
            raise ValueError("missing baseline prediction for training example")
        reference = baseline[row["id"]]
        for key in ("kind", "context", "question", "answers"):
            if row[key] != reference[key]:
                raise ValueError(f"baseline/training mismatch in {key}")
        if row["kind"] == "answerable" and reference["em"] == 1:
            anchors.add(row["id"])
    if not anchors:
        raise ValueError("no exact-correct supported training anchors")
    return anchors
