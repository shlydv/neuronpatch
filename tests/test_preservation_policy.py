"""The experiment must be allowed to reject every patch."""

import runpy
from pathlib import Path

import pytest

choose = runpy.run_path(str(Path(__file__).parents[1] / "experiments/preservation_qa/policy.py"))[
    "choose_preserving_candidate"
]


def record(answerable, missing, sentinel=1.0, lr=0.001):
    return {
        "validation": {
            "answerable": answerable,
            "missing": missing,
            "accuracy": (answerable + missing) / 2,
        },
        "sentinel": {"accuracy": sentinel},
        "lr": lr,
    }


def test_rejects_aggregate_gain_that_breaks_preserved_answers():
    assert choose(record(1, 0.5), {"bad": record(0.8, 1)}) is None


def test_rejects_no_improvement_and_sentinel_damage():
    assert choose(record(1, 0.5), {"same": record(1, 0.5), "damage": record(1, 1, 0.8)}) is None


def test_chooses_eligible_candidate_and_lower_rate_tie():
    assert choose(record(1, 0.5), {"a": record(1, 1), "b": record(1, 1, lr=0.0003)}) == "b"
    assert choose(record(1, 0.5), {}) is None


def test_followup_entities_are_disjoint_from_previous_pilot():
    import json

    root = Path(__file__).parents[1]
    pytest.importorskip("peft")
    new_dataset = runpy.run_path(str(root / "experiments/preservation_qa/run.py"))["dataset"]
    old = json.loads((root / "experiments/grounded_qa/results/data.json").read_text())
    seen = {r["entity"] for rows in old.values() for r in rows if "entity" in r}
    for split, count in (("train", 96), ("cal", 24), ("val", 48), ("test", 96)):
        rows = new_dataset(split, count)
        entities = {r["entity"] for r in rows}
        assert not seen & entities
        seen |= entities
        assert len(rows) == count
        assert sum(r["kind"] == "missing" for r in rows) == count // 2
        assert all(r["answer"] in r["prompt"] for r in rows if r["kind"] == "answerable")
