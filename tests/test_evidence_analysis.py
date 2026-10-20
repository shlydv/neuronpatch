"""Check metric groups and bias-invariant diagnostics without model fitting."""

import pytest

from experiments.evidence_pairs.analyze import counts, discrimination, preserve


def test_missing_accuracy_and_preservation_denominators_are_separate():
    base = [
        {"id": "s", "role": "primary", "kind": "answerable", "em": 1},
        {"id": "m", "role": "primary", "kind": "missing", "em": 0},
        {"id": "i", "role": "invariant", "kind": "answerable", "em": 1},
    ]
    edited = [{**row, "em": float(row["id"] != "s")} for row in base]
    assert counts(edited)["missing"] == {"correct": 1, "n": 1}
    assert preserve(edited, base, "supported") == {"retained": 0, "n": 1, "fraction": 0}
    assert preserve(edited, base, "invariant")["fraction"] == 1
    utility = [{"id": "u", "em": 1}]
    assert preserve(utility, utility)["fraction"] == 1


def test_discrimination_ignores_control_questions_and_shared_refusal_bias():
    rows = [
        {"id": "s", "role": "primary", "kind": "answerable", "pair_id": "p"},
        {"id": "m", "role": "primary", "kind": "missing", "pair_id": "p"},
        {"id": "i", "role": "invariant", "kind": "answerable", "pair_id": "p"},
    ]
    signals = [
        {"id": "s", "unknown_log_odds": -2.0},
        {"id": "m", "unknown_log_odds": -1.0},
        {"id": "i", "unknown_log_odds": 100.0},
    ]
    original = discrimination(rows, signals)
    shifted = discrimination(
        rows, [{**s, "unknown_log_odds": s["unknown_log_odds"] + 9} for s in signals]
    )
    assert original == shifted
    assert original["unknown_log_odds_auroc"] == pytest.approx(1)
    assert original["mean_paired_gap"] == pytest.approx(1)
