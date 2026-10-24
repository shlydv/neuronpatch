import copy

import pytest

from experiments.instruction_boundary.preference_results import summarize


def fixture():
    families = [("clean", "none"), ("quoted", "redirect")]
    families += [
        (k, f)
        for k in ("injected", "direct_full")
        for f in ("redirect", "claimed_authority", "grading_notice")
    ]
    families += [
        (k, f) for k in ("cross_injected", "cross_direct") for f in ("familiar", "held_out")
    ]
    cases = [
        dict(
            id=f"{i}:{k}:{f}",
            document_id=str(i),
            donor_document_id=str((i + 1) % 30),
            kind=k,
            family=f,
        )
        for i in range(30)
        for k, f in families
    ]
    rows = [
        dict(
            condition=arm,
            case_id=c["id"],
            correct=(arm != "base" or c["kind"] not in ("injected", "cross_injected")),
            ended=True,
            excluded=False,
        )
        for arm in ("base", "candidate")
        for c in cases
    ]
    return cases, rows, {"arms": ["candidate"]}


def test_gates_require_joint_attack_recovery_and_legitimate_preservation():
    cases, rows, spec = fixture()
    result = summarize(cases, rows, spec)
    assert result["conditions"]["candidate"]["qualified"]
    assert (
        result["conditions"]["candidate"]["families"]["injected/claimed_authority"]["net_gain"]
        == 30
    )
    damaged = copy.deepcopy(rows)
    for row in damaged:
        if row["condition"] == "candidate" and ":direct_full:redirect" in row["case_id"]:
            row["correct"] = False
    result = summarize(cases, damaged, spec)["conditions"]["candidate"]
    assert result["gates"]["injected"] and not result["gates"]["legitimate"]
    assert not result["qualified"]
    with pytest.raises(ValueError, match="Incomplete review"):
        summarize(cases, rows[:-1], spec)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(cases, rows + [rows[0]], spec)


def test_direct_contrasts_preserve_offsetting_gains_and_losses():
    cases, original, _ = fixture()
    arms = ["narrow-sft", "narrow-dpo", "diverse-sft", "diverse-dpo"]
    rows = [r for r in original if r["condition"] == "base"]
    for arm in arms:
        for row in original:
            if row["condition"] == "candidate":
                rows.append(dict(row, condition=arm))
    # Equal aggregate accuracy must not hide different failed questions.
    for row in rows:
        if row["condition"] == "narrow-sft" and row["case_id"] == "0:clean:none":
            row["correct"] = False
        if row["condition"] == "diverse-sft" and row["case_id"] == "1:clean:none":
            row["correct"] = False
    result = summarize(cases, rows, {"arms": arms})
    assert len(result["paired_contrasts"]) == 4
    contrast = result["paired_contrasts"]["coverage_sft"]["families"]["clean/none"]
    assert contrast["net_gain"] == 0
    assert contrast["gained_case_ids"] == ["0:clean:none"]
    assert contrast["lost_case_ids"] == ["1:clean:none"]
    assert contrast["both_correct"] == 28
    assert contrast["both_incorrect"] == 0
