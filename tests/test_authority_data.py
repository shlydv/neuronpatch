import copy

import pytest

from experiments.instruction_boundary.authority_data import fresh_confirmation
from experiments.instruction_boundary.prepare import build, context_key


def test_fresh_confirmation_keeps_fitting_banks_and_excludes_every_prior_document():
    rows = [{"context": f"table {i}", "question": "value?", "ideal": str(i)} for i in range(10)]
    previous = build(rows[:5], {"train": 2, "calibration": 1, "confirmation": 2})
    previous["excluded_documents"] = [context_key(rows[5]["context"])]
    before = copy.deepcopy(previous)
    result = fresh_confirmation(rows, previous, count=3)
    assert result["splits"]["train"] == previous["splits"]["train"]
    assert result["splits"]["calibration"] == previous["splits"]["calibration"]
    held = {r["document_id"] for r in result["splits"]["confirmation"]}
    assert len(held) == 3 and not held.intersection(result["excluded_confirmation_documents"])
    assert len(result["splits"]["confirmation"]) == 33
    assert result == fresh_confirmation(list(reversed(rows)), previous, count=3)
    assert previous == before
    for row in result["splits"]["confirmation"]:
        if row["kind"] == "direct_full":
            assert row["expected"] == row["marker"]
            assert row["messages"][-1]["content"].endswith(row["marker"] + ".")
    with pytest.raises(ValueError, match="Insufficient"):
        fresh_confirmation(rows, previous, count=5)
