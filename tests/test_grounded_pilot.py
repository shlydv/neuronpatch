"""Guard the pilot's split and missing-evidence construction."""

import runpy
from pathlib import Path

import pytest

pytest.importorskip("peft")
PILOT = runpy.run_path(str(Path(__file__).parents[1] / "experiments/grounded_qa/run.py"))


def test_grounded_data_has_no_split_leakage_and_balanced_pairs():
    seen = set()
    for split, count in (("train", 64), ("cal", 16), ("val", 24), ("test", 48)):
        rows = PILOT["dataset"](split, count)
        entities = {r["entity"] for r in rows}
        assert not seen & entities
        seen |= entities
        assert len(rows) == count
        for entity in entities:
            pair = [r for r in rows if r["entity"] == entity]
            assert {r["kind"] for r in pair} == {"answerable", "missing"}
            assert (
                pair[0]["prompt"].split("\nQuestion:")[0]
                == pair[1]["prompt"].split("\nQuestion:")[0]
            )
            for row in pair:
                record, question = row["prompt"].split("\nQuestion: What is the ")
                field = question.rstrip("?")
                if row["kind"] == "answerable":
                    assert row["answer"] in record
                    assert f"{field}:" in record
                else:
                    assert f"{field}:" not in record
                    assert row["answer"] == "UNKNOWN"
