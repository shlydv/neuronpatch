"""Literal labels and counterbalancing for the prospective relation probes."""

from experiments.relation_contrast.prepare import build


def test_literal_ground_truth_and_counterbalancing():
    rows = build()
    by_id = {row["id"]: row for row in rows}
    assert len(rows) == len(by_id) == 96
    for family in ("tournament", "interview", "grant", "release"):
        target = by_id[f"{family}:0:0:target_last"]
        assert target["answer_groups"] == [["Oren"]]
        assert by_id[f"{family}:0:1:target_last"]["answer_groups"] == [["Mira"]]
        assert by_id[f"{family}:0:0:missing"]["answer_groups"] == [["UNKNOWN"]]
        assert "Oren" not in by_id[f"{family}:0:0:missing"]["messages"][-1]["content"]
        assert "Mira" not in by_id[f"{family}:0:0:target_only"]["messages"][-1]["content"]
    assert (
        "defeated Mira to reach the semifinal"
        in by_id["tournament:0:0:target_last"]["messages"][-1]["content"]
    )
    assert "Arlen defeated Oren" in by_id["tournament:0:0:target_last"]["messages"][-1]["content"]
    for row in rows:
        if row["condition"] == "target_last":
            other = by_id[row["id"].replace("target_last", "target_first")]
            a = row["messages"][-1]["content"].split("\n\nQuestion:")[0]
            b = other["messages"][-1]["content"].split("\n\nQuestion:")[0]

            def documents(text):
                return sorted(x.split("] ", 1)[1] for x in text.splitlines() if "] " in x)

            assert documents(a) == documents(b)
            assert row["answer_groups"] == other["answer_groups"]
