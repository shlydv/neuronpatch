import pytest

from experiments.instruction_boundary.preference_review import packet, review_cases
from experiments.instruction_boundary.prepare import examples


def test_trusted_replacement_is_reviewed_against_donor_not_original_question():
    a = examples({"context": "A 7", "question": "A?", "ideal": "7"}, "a")[0]
    b = examples({"context": "B 8", "question": "B?", "ideal": "8"}, "b")[0]
    inside = {
        **a,
        "id": "inside",
        "kind": "cross_injected",
        "donor_document_id": "b",
        "alternative": "8",
    }
    outside = {
        **inside,
        "id": "outside",
        "kind": "cross_direct",
        "expected": "8",
        "alternative": "7",
    }
    cases = [a, b, inside, outside]
    outputs = {
        "base": [
            {
                "id": c["id"],
                "text": ("8" if c["id"] in ("inside", "outside") else c["expected"]),
                "ended": True,
            }
            for c in cases
        ]
    }
    docs, key = packet(cases, outputs)
    indexed = {d["document_id"]: d for d in docs}
    assert indexed["a"]["reference_answer"] == "7"
    assert indexed["b"]["reference_answer"] == "8"
    assert any(r["text"] == "8" for r in indexed["a"]["responses"])
    assert any(r["text"] == "8" for r in indexed["b"]["responses"])
    refs = {r["case_id"]: r["response_id"] for r in key["attribution"]}
    assert refs["inside"] != refs["outside"]
    assert outside["document_id"] == "a"  # No mutation of original pairing metadata.
    with pytest.raises(ValueError, match="target differs"):
        review_cases([a, b, {**outside, "expected": "7"}])
