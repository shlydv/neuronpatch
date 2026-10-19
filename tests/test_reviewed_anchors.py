import copy
import runpy
from pathlib import Path

import pytest

module = runpy.run_path(str(Path(__file__).parents[1] / "experiments/real_qa/reviewed_anchors.py"))
select = module["reviewed_additions"]
digest = module["evidence_digest"]


def fixture():
    row = dict(
        id="one",
        kind="answerable",
        context="Paris is the capital.",
        question="What is the capital?",
        answers=["Paris"],
        output="It is Paris.",
        em=0,
        abstained=False,
    )
    review = dict(
        reviews=[
            dict(id="one", category="valid_answer_extra_wording", evidence_sha256=digest(row))
        ],
        eligible_additional_ids=["one"],
    )
    return row, review


def test_review_is_bound_to_training_scope_and_actual_baseline_output():
    row, review = fixture()
    assert select([row], [row], review) == {"one"}
    with pytest.raises(ValueError, match="outside training"):
        select([], [row], review)
    with pytest.raises(ValueError, match="evidence changed"):
        select([row], [row | {"output": "It is not Paris."}], review)
    with pytest.raises(ValueError, match="evidence mismatch"):
        select([row | {"context": "London is the capital."}], [row], review)


def test_rejected_reviews_cannot_be_silently_added_and_duplicates_fail():
    row, review = fixture()
    changed = copy.deepcopy(review)
    changed["reviews"][0]["category"] = "unsupported_reframing"
    with pytest.raises(ValueError, match="disagree"):
        select([row], [row], changed)
    with pytest.raises(ValueError, match="duplicate"):
        select([row], [row], review | {"eligible_additional_ids": ["one", "one"]})


def test_exact_or_refused_answers_are_not_nonexact_additions():
    row, review = fixture()
    for change in ({"em": 1}, {"abstained": True}, {"kind": "missing", "answers": []}):
        changed = row | change
        updated = copy.deepcopy(review)
        updated["reviews"][0]["evidence_sha256"] = digest(changed)
        with pytest.raises(ValueError, match="supported nonexact nonrefusal"):
            select([changed], [changed], updated)
