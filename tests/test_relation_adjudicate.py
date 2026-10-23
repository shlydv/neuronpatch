import pytest

from experiments.relation_learning.adjudicate import adjudicate, digest


def fixture():
    case = {"utility_test": [{"id": "binary", "answers": ["2"]}]}

    def answer(text):
        return {
            "id": "binary",
            "text": text,
            "ended": True,
            "truncated": False,
            "unexpected_special": False,
        }

    predictions = {
        "base": {"utility_test": [answer("The answer is 2.")]},
        "patch": {"utility_test": [answer("10")]},
    }
    decisions = [
        {
            "model": model,
            "id": "binary",
            "response_sha256": digest(predictions[model]["utility_test"][0]["text"]),
            "correct": correct,
            "reason": "Explanatory correct answer" if correct else "Incorrect decimal value",
        }
        for model, correct in (("base", True), ("patch", False))
    ]
    return case, predictions, decisions


def test_semantic_base_denominator_exposes_regression_hidden_by_exact_matching():
    case, predictions, decisions = fixture()
    result = adjudicate(case, predictions, decisions)
    patch = next(r for r in result["summary"] if r["model"] == "patch")
    assert patch["base_correct"] == 1 and patch["correct"] == 0
    assert patch["regressions"] == 1 and patch["corrections"] == 0


def test_partial_or_stale_review_cannot_produce_summary():
    case, predictions, decisions = fixture()
    with pytest.raises(ValueError, match="incomplete"):
        adjudicate(case, predictions, decisions[:1])
    predictions["patch"]["utility_test"][0]["text"] = "2"
    with pytest.raises(ValueError, match="different response"):
        adjudicate(case, predictions, decisions)


@pytest.mark.parametrize("failure", ["empty", "truncated", "unexpected_special"])
def test_structural_failure_cannot_be_overridden_as_correct(failure):
    case, predictions, decisions = fixture()
    output = predictions["patch"]["utility_test"][0]
    if failure == "empty":
        output["text"] = ""
    else:
        output[failure] = True
    decisions[1].update(correct=True, response_sha256=digest(output["text"]))
    with pytest.raises(ValueError, match="cannot pass"):
        adjudicate(case, predictions, decisions)


def test_complete_reference_match_needs_no_manual_override():
    case, predictions, decisions = fixture()
    predictions["patch"]["utility_test"][0]["text"] = "2"
    result = adjudicate(case, predictions, decisions[:1])
    assert all(r["correct"] for r in result["records"])
    assert all(r["corrections"] == r["regressions"] == 0 for r in result["summary"])
