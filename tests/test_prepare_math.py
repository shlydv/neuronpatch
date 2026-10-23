from decimal import Decimal

from experiments.relation_learning.prepare_math import final_number, partition


def test_parser_does_not_mistake_reasoning_or_conflicting_answers_for_final():
    assert Decimal(final_number("Reasoning 5\n#### 1,234.50")) == Decimal("1234.5")
    assert final_number("#### -0.5") == "-0.5"
    for text in (
        "We got 42",
        "#### 2 or 3",
        "#### 1\n#### 2",
        "#### 1,2",
        "x #### 2",
        "#### 2\nActually 3",
        "#### NaN",
    ):
        assert final_number(text) is None


def test_partition_is_order_independent_disjoint_and_removes_duplicate_groups():
    rows = [{"question": f"problem {i}", "answer": f"A <<1+1=2>>\n#### {i}"} for i in range(10)]
    rows += [{"question": "PROBLEM  1", "answer": "#### 999"}]
    sizes = {"train": 4, "calibration": 2, "development": 2}
    result = partition(rows, sizes)
    assert result == partition(list(reversed(rows)), sizes)
    assert result["excluded_duplicate_rows"] == 2
    selected = [r for split in result["splits"].values() for r in split]
    assert len({r["id"] for r in selected}) == 8
    assert all("<<" not in r["response"] for r in selected)
    assert all(r["question"] not in ("problem 1", "PROBLEM  1") for r in selected)
