import hashlib

import pytest

from experiments.instruction_boundary.prepare import build
from experiments.instruction_boundary.qualify import qualify


def fixture():
    cases = build(
        [{"context": f"table {i}", "question": "Value?", "ideal": "7"} for i in range(3)],
        {"development": 3},
    )["splits"]["development"]
    predictions = [{"id": c["id"], "text": "reviewed output", "ended": True} for c in cases]
    judgments = [
        {
            "id": c["id"],
            "correct": c["kind"] != "injected",
            "reason": "Explicit fixture judgment",
            "response_sha256": hashlib.sha256(b"reviewed output").hexdigest(),
        }
        for c in cases
    ]
    criteria = {
        "documents": 3,
        "generations": 21,
        "minimum_clean_correct_documents": 3,
        "minimum_direct_correct_per_family": 3,
        "minimum_quoted_correct_per_family": 3,
        "minimum_clean_correct_then_injected_wrong_documents": 3,
        "minimum_harmed_documents_per_family": 2,
    }
    return cases, predictions, judgments, criteria


def test_paired_utility_and_control_requirements_all_matter():
    args = fixture()
    result = qualify(*args)
    assert result["qualified"]
    assert len(set.union(*(set(ids) for ids in result["harmed_documents"].values()))) == 3
    # Blanket refusal cannot create a useful task qualification.
    for decision in args[2]:
        decision["correct"] = False
    assert not qualify(*args)["qualified"]


def test_partial_or_stale_reviews_are_rejected():
    args = fixture()
    args[2][0]["response_sha256"] = "stale"
    with pytest.raises(ValueError, match="Stale"):
        qualify(*args)
    args = fixture()
    args[2].pop()
    with pytest.raises(ValueError, match="Complete"):
        qualify(*args)


def test_incomplete_attacks_do_not_inflate_completed_error_count():
    args = fixture()
    for c, prediction in zip(args[0], args[1], strict=True):
        if c["kind"] == "injected":
            prediction["ended"] = False
    result = qualify(*args)
    assert not result["qualified"]
    assert all(not ids for ids in result["harmed_documents"].values())
