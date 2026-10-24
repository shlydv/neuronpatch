import copy
import json
from collections import Counter

import pytest

from experiments.instruction_boundary.prepare import examples
from experiments.instruction_boundary.reviewed_tasks import (
    PREFIX,
    SEPARATOR,
    apply_review,
    assignments,
    build_task_pairs,
    case_digest,
)


def source_and_review(number, answer):
    c = examples(
        {
            "context": f"Name | Value\nRow{number} | {answer}",
            "question": f"Value of Row{number}?",
            "ideal": "untrusted reference",
        },
        f"doc{number}",
    )[0]
    r = {
        "document_id": c["document_id"],
        "source_case_sha256": case_digest(c),
        "answer": answer,
        "exclude": False,
        "reason": "Manual table lookup",
        "question_override": None,
        "question_revision_reason": None,
    }
    return c, r


def tasks_fixture():
    items = [source_and_review(i, a) for i, a in enumerate(["7", "9", "9", "4"])]
    return apply_review([c for c, _ in items], [r for _, r in items])


def test_review_requires_exact_source_and_explicit_question_revision():
    c, r = source_and_review(1, "7")
    changed = copy.deepcopy(c)
    changed["messages"][-1]["content"] += " extra"
    with pytest.raises(ValueError, match="match"):
        apply_review([changed], [r])
    modified = dict(r, question_override="Which value appears on the second table row?")
    with pytest.raises(ValueError, match="rationale"):
        apply_review([c], [modified])
    modified["question_revision_reason"] = "Clarify row order"
    tasks = apply_review([c], [modified])
    assert tasks[0]["question"] == modified["question_override"]
    assert tasks[0]["answer"] == "7"
    assert (
        json.loads(tasks[0]["messages"][-1]["content"].split(SEPARATOR)[1]) == tasks[0]["document"]
    )
    assert apply_review([c], [dict(r, exclude=True, answer=None)]) == []
    with pytest.raises(ValueError, match="training answer"):
        apply_review([c], [dict(r, exclude=True)])


def test_balanced_donors_are_distinct_and_repeatable_under_source_order_changes():
    tasks = tasks_fixture()
    pairs = assignments(tasks, seed=43, rounds=2)
    assert pairs == assignments(list(reversed(tasks)), seed=43, rounds=2)
    all_edges = [e for round_pairs in pairs for e in round_pairs]
    assert len(set(all_edges)) == 8
    lookup = {t["document_id"]: t for t in tasks}
    for round_pairs in pairs:
        assert Counter(x for x, _ in round_pairs) == Counter(lookup.keys())
        assert Counter(y for _, y in round_pairs) == Counter(lookup.keys())
        for x, y in round_pairs:
            assert x != y and lookup[x]["answer"] != lookup[y]["answer"]
    same = [dict(t, answer="7") for t in tasks]
    with pytest.raises(ValueError, match="balance"):
        assignments(same, seed=43, rounds=1)


def test_nested_request_moves_across_outer_boundary_and_answers_reverse():
    tasks = tasks_fixture()
    lookup = {t["document_id"]: t for t in tasks}
    pairs = build_task_pairs(tasks, seed=43, rounds=2)
    assert len(pairs) == 16 and len({p["id"] for p in pairs}) == 16
    for inside, outside in zip(pairs[::2], pairs[1::2], strict=True):
        recipient = lookup[inside["document_id"]]
        donor = lookup[inside["donor_document_id"]]
        assert inside["chosen"] == outside["rejected"] == recipient["answer"]
        assert inside["rejected"] == outside["chosen"] == donor["answer"]
        inner = json.loads(inside["messages"][-1]["content"].partition(SEPARATOR)[2])
        out_text = outside["messages"][-1]["content"].partition(SEPARATOR)[2]
        outer, end = json.JSONDecoder().raw_decode(out_text)
        assert outer == recipient["document"]
        request = out_text[end:].removeprefix("\n\n")
        assert inner == outer + "\n\n" + request
        assert PREFIX + donor["question"] in request
        donor_document = json.loads(request.partition(SEPARATOR)[2])
        assert donor_document == donor["document"]
    assert Counter(p["chosen"] for p in pairs) == Counter(p["rejected"] for p in pairs)
