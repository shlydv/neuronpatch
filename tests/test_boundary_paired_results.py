import copy

import pytest

from experiments.instruction_boundary.paired_results import comparison, summarize


def test_discordant_pairs_and_identity_checks():
    left = {str(i): True for i in range(6)}
    right = {str(i): False for i in range(6)}
    assert comparison(left, right)["discordant_pair_p_two_sided"] == 0.03125
    assert comparison(left, left)["ties"] == 6
    assert comparison(left, left)["discordant_pair_p_two_sided"] == 1
    with pytest.raises(ValueError, match="identical"):
        comparison(left, {"other": True})


def test_recovery_denominator_and_clean_regressions_are_separate():
    cases = [
        {
            "id": f"{doc}:{kind}",
            "document_id": doc,
            "kind": kind,
            "family": "none" if kind == "clean" else "redirect",
        }
        for doc in ("a", "b", "c")
        for kind in ("clean", "injected", "direct_full")
    ]
    correctness = {
        "base": [True, False, True, False, False, True, True, True, True],
        "candidate": [False, True, True, True, True, True, True, False, False],
    }
    records = [
        {
            "condition": name,
            "case_id": case["id"],
            "correct": correct,
            "excluded": False,
            "ended": True,
        }
        for name, values in correctness.items()
        for case, correct in zip(cases, values, strict=True)
    ]
    result = summarize(cases, records)
    candidate = result["conditions"]["candidate"]
    assert result["baseline_clean_correct_documents"] == ["a", "c"]
    assert candidate["clean:none"]["correct"] == 2  # Same aggregate as baseline.
    assert candidate["clean:none"]["regressions"] == ["a"]
    assert candidate["injected:redirect"]["baseline_paired_failures"] == ["a"]
    assert candidate["injected:redirect"]["recovered"] == ["a"]  # Not b: base could not solve it.
    paired = result["paired_comparisons"]["redirect"]["candidate"]["base"]
    assert paired["wins"] == ["a"] and paired["losses"] == ["c"]
    assert paired["net_correct"] == 0
    assert candidate["direct_full:redirect"]["regressions"] == ["c"]
    corrupt = copy.deepcopy(records)
    corrupt[0]["excluded"] = True
    with pytest.raises(ValueError, match="document-wide"):
        summarize(cases, corrupt)
    with pytest.raises(ValueError, match="Complete"):
        summarize(cases, records[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(cases, records + records[:1])
