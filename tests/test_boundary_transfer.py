from copy import deepcopy

import pytest

from experiments.instruction_boundary.prepare import build
from experiments.instruction_boundary.transfer_data import fresh_documents, teacher_supervision


def rows():
    return [
        {"context": f"table {i}", "question": "Which value?", "ideal": str(i)} for i in range(12)
    ]


def test_fresh_documents_exclude_all_old_banks_and_ignore_source_order():
    old = build(rows(), {"train": 2, "calibration": 1, "development": 2})
    excluded = [r["document_id"] for r in old["membership"]]
    sizes = {"train": 2, "calibration": 1, "confirmation": 2}
    new = fresh_documents(rows(), excluded, sizes)
    assert new == fresh_documents(list(reversed(rows())), excluded, sizes)
    assert not set(excluded) & {r["document_id"] for r in new["membership"]}


def test_teacher_targets_do_not_use_gold_or_held_out_instruction_family():
    cases = build(rows(), {"train": 2})["splits"]["train"]
    before = deepcopy(cases)
    clean = [r for r in cases if r["kind"] == "clean"]
    predictions = [
        {"id": clean[0]["id"], "text": "A possibly wrong teacher answer", "ended": True},
        {"id": clean[1]["id"], "text": "Unfinished answer", "ended": False},
    ]
    result = teacher_supervision(cases, predictions)
    assert cases == before
    assert len(result["task"]) == 1 and len(result["preserve"]) == 3
    assert result["task"][0]["target"] == predictions[0]["text"]
    assert result["excluded_documents"] == [clean[1]["document_id"]]
    assert all("claimed_authority" not in row["id"] for row in result["task"] + result["preserve"])
    for row in cases:
        row["expected"] = "POISONED REFERENCE"
    assert teacher_supervision(cases, predictions) == result
    with pytest.raises(ValueError):
        teacher_supervision(cases, predictions[:1])
    with pytest.raises(ValueError):
        teacher_supervision(cases, predictions + predictions[:1])
