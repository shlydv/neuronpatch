"""Adversarial checks for frozen decision rules, without model outputs."""

from copy import deepcopy

import pytest

from experiments.instruction_boundary.supervision_verdict import ARMS, FAMILIES, evaluate


def fixture():
    cases = []
    for document in range(32):
        for kind, family in [("clean", "none")] + [
            (kind, family)
            for kind in ("injected", "direct", "quoted", "direct_full")
            for family in FAMILIES
        ]:
            cases.append(
                {
                    "id": f"{document}:{kind}:{family}",
                    "document_id": str(document),
                    "kind": kind,
                    "family": family,
                }
            )
    records = []
    for condition in ["base", *ARMS, *("reused-" + arm for arm in ARMS)]:
        for case in cases:
            correct = True
            if case["kind"] == "direct_full" and condition.startswith("reused-"):
                correct = False
            if case["kind"] == "injected":
                correct = condition != "base"
                if "random_grouped" in condition:
                    correct = int(case["document_id"]) >= 4
            records.append(
                {
                    "condition": condition,
                    "case_id": case["id"],
                    "correct": correct,
                    "ended": True,
                    "excluded": False,
                }
            )
    return cases, records


def test_absolute_repair_and_secondary_gates():
    cases, records = fixture()
    result = evaluate(cases, records)
    assert all(row["status"] == "pass" for row in result["repair"].values())
    assert result["secondary_selection"]["status"] == "pass"
    # A high average cannot hide a failure on one family.
    for row in records:
        if row["condition"] == ARMS[0] and row["case_id"].endswith("direct_full:grading_notice"):
            if int(row["case_id"].split(":")[0]) < 3:
                row["correct"] = False
    result = evaluate(cases, records)
    assert result["repair"][ARMS[0]]["status"] == "fail"
    assert result["secondary_selection"]["status"] == "not_eligible"


def test_net_corrections_not_only_new_successes():
    cases, records = fixture()
    for row in records:
        if row["condition"] == "reused-" + ARMS[0] and row["case_id"].endswith(
            "direct_full:redirect"
        ):
            row["correct"] = int(row["case_id"].split(":")[0]) < 9
    result = evaluate(cases, records)["repair"][ARMS[0]]
    assert result["redirect_net_corrections"] == 23
    assert result["status"] == "fail"


def test_no_certification_with_insufficient_base_ability():
    cases, records = fixture()
    for row in records:
        if row["condition"] == "base" and row["case_id"].endswith("clean:none"):
            row["correct"] = int(row["case_id"].split(":")[0]) < 15
    result = evaluate(cases, records)
    assert all(r["status"] == "insufficient_headroom" for r in result["repair"].values())
    assert result["secondary_selection"]["status"] == "not_eligible"


def test_missing_cells_and_incomplete_correct_answers_rejected():
    cases, records = fixture()
    with pytest.raises(ValueError, match="complete frozen case grid"):
        evaluate(cases[:-1], records)
    with pytest.raises(ValueError, match="coverage"):
        evaluate(cases, records[:-1])
    bad = deepcopy(records)
    bad[0]["ended"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate(cases, bad)


def test_retention_is_compared_to_own_old_endpoint():
    cases, records = fixture()
    for row in records:
        if row["condition"] == ARMS[0] and row["case_id"].endswith("injected:redirect"):
            if int(row["case_id"].split(":")[0]) < 2:
                row["correct"] = False
    result = evaluate(cases, records)
    assert not result["repair"][ARMS[0]]["checks"]["task_retention:redirect"]
    assert result["repair"][ARMS[1]]["status"] == "pass"
