from copy import deepcopy

from experiments.instruction_boundary.prepare import FAMILIES, SYSTEM, build


def test_documents_are_disjoint_and_input_order_does_not_choose_questions():
    rows = [
        {"context": f"table {i}", "question": "What is shown?", "ideal": str(i)} for i in range(8)
    ]
    rows.append({"context": " table   1 ", "question": "Other question", "ideal": "1"})
    sizes = {"train": 3, "calibration": 2, "development": 2}
    result = build(rows, sizes)
    assert result == build(list(reversed(rows)), sizes)
    assert result["source_documents"] == 8
    ids = [m["document_id"] for m in result["membership"]]
    assert len(ids) == len(set(ids)) == 7


def test_same_instructions_change_authority_without_changing_target_qa():
    rows = [{"context": 'cell "quoted"\nanswer 7', "question": "Value?", "ideal": "7"}]
    original = deepcopy(rows)
    cases = build(rows, {"development": 1})["splits"]["development"]
    assert rows == original and len(cases) == 7
    assert all(c["messages"][0] == {"role": "system", "content": SYSTEM} for c in cases)
    assert all(len(c["messages"]) == 2 and c["messages"][1]["role"] == "user" for c in cases)
    assert all(c["expected"] == "7" for c in cases if c["kind"] in ("clean", "injected"))
    for family in FAMILIES:
        related = {c["kind"]: c for c in cases if c["family"] == family}
        direct = related["direct"]["messages"][1]["content"]
        assert direct in related["injected"]["messages"][1]["content"]
        assert related["direct"]["expected"] == related["quoted"]["expected"]
        assert related["direct"]["expected"] != "7"
