import runpy
from pathlib import Path

import pytest

select = runpy.run_path(str(Path(__file__).parents[1] / "experiments/real_qa/training_data.py"))[
    "exact_correct_anchors"
]


def row(identifier, kind="answerable", correct=True):
    return {
        "id": identifier,
        "kind": kind,
        "em": float(correct),
        "context": "a passage",
        "question": "a question",
        "answers": ["a value"] if kind == "answerable" else [],
    }


def test_anchors_exclude_outside_prefix_unsupported_and_incorrect_examples():
    training = [row("correct"), row("wrong", correct=False), row("missing", kind="missing")]
    baseline = training + [row("outside_this_training_prefix")]
    assert select(training, baseline) == {"correct"}


def test_missing_or_mismatched_baseline_fails_instead_of_silent_anchor_loss():
    with pytest.raises(ValueError, match="missing baseline"):
        select([row("one"), row("two")], [row("one")])
    with pytest.raises(ValueError, match="mismatch"):
        select([row("one")], [row("one") | {"context": "changed evidence"}])
    with pytest.raises(ValueError, match="duplicate baseline"):
        select([row("one")], [row("one"), row("one")])
