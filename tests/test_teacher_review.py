from copy import deepcopy

import pytest

from experiments.instruction_boundary.teacher_review import approved_teachers, review_identity


def fixture():
    inventory = {
        "train": [
            {
                "task": {"document_id": did, "document": "source table"},
                "teacher": {"id": f"{did}:clean:none", "tokens": [1, 2], "text": text},
            }
            for did, text in [("a", "correct"), ("b", "incorrect")]
        ]
    }
    decisions = [
        {
            "split": "train",
            "document_id": r["task"]["document_id"],
            "identity": review_identity(r),
            "verdict": verdict,
            "rationale": "Checked against source",
            "reviewer": "test fixture",
        }
        for r, verdict in zip(inventory["train"], ("accept", "reject"), strict=True)
    ]
    return inventory, decisions


def test_complete_review_excludes_wrong_teacher_without_losing_receipt():
    inventory, decisions = fixture()
    selected, receipt = approved_teachers(inventory, decisions)
    assert [r["task"]["document_id"] for r in selected["train"]] == ["a"]
    assert receipt == decisions
    selected["train"][0]["teacher"]["tokens"][0] = 9
    assert inventory["train"][0]["teacher"]["tokens"] == [1, 2]
    with pytest.raises(ValueError, match="1 pending"):
        approved_teachers(inventory, decisions[:1])


@pytest.mark.parametrize("field", ["document", "tokens", "text"])
def test_changed_source_or_teacher_requires_new_review(field):
    inventory, decisions = fixture()
    record = inventory["train"][0]
    record["task" if field == "document" else "teacher"][field] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        approved_teachers(inventory, decisions)


def test_duplicate_unknown_and_cross_split_reviews_fail():
    inventory, decisions = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        approved_teachers(inventory, decisions + decisions[:1])
    bad = deepcopy(decisions)
    bad[0]["document_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        approved_teachers(inventory, bad)
    inventory["calibration"] = inventory["train"][:1]
    with pytest.raises(ValueError, match="cross-split"):
        approved_teachers(inventory, decisions)
