import runpy
from pathlib import Path

import pytest

compare = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/real_qa/compare_predictions.py")
)["compare"]


def rows():
    return [
        dict(
            id=f"{article}-{kind}-{i}",
            article=article,
            kind=kind,
            context="text",
            question="question",
            answers=["answer"],
            em=0,
            f1=0,
            abstained=False,
        )
        for article in ("a", "b")
        for kind in ("answerable", "missing")
        for i in range(4)
    ]


def test_identity_has_zero_delta_and_duplicate_or_changed_evidence_fails():
    before = rows()
    result = compare(before, list(reversed(before)))
    assert all(
        m == {"after_minus_before": 0.0, "percentile_95": [0.0, 0.0]}
        for m in result["metrics"].values()
    )
    with pytest.raises(ValueError, match="unique and match"):
        compare(before, before + [before[0]])
    with pytest.raises(ValueError, match="evidence"):
        compare(before, [before[0] | {"context": "changed"}] + before[1:])


def test_correlated_questions_resample_whole_articles_not_individual_questions():
    before = [r | {"em": int(r["article"] == "b")} for r in rows()]
    after = [r | {"em": int(r["article"] == "a")} for r in rows()]
    result = compare(before, after)
    # Each article's four questions move together. Resampling both articles can
    # select a/a or b/b, giving +/-1 despite eight questions in each category.
    assert result["metrics"]["supported_em"] == {
        "after_minus_before": 0.0,
        "percentile_95": [-1.0, 1.0],
    }
