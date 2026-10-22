"""Compare both frozen deployment states without selecting a new edit."""

import argparse
import json
from pathlib import Path

from experiments.paired_mlp.data import whole_hit


def summarize(parent, results):
    protocol = json.loads((results / "protocol.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["new_fits"] == 0
    cases = {c["case"]: c for c in json.loads((parent / "inputs.json").read_text())["cases"]}
    records = json.loads((results / "endpoints.json").read_text())
    assert len(records) == len(protocol["endpoints"])
    rows = []
    for record in records:
        assert record["parent_merged_teacher_exact"] and record["parent_merged_generation_exact"]
        c, name = record["case"], record["condition"]["id"]
        case = cases[c]
        baseline = json.loads((parent / f"case-{c}-base.json").read_text())
        states = {}
        for state in ("live", "merged"):
            answers = json.loads((results / f"case-{c}-{name}-{state}.json").read_text())
            banks = {}
            for bank in protocol["generation_banks"]:
                before = baseline[bank]
                after = answers[bank]
                assert len(before) == len(after) == len(case[bank])
                hits, lost, mentions = [], [], []
                for row, b, a in zip(case[bank], before, after, strict=True):
                    hit = whole_hit(a, row["answers"])
                    if hit:
                        hits.append(row["id"])
                    if whole_hit(b, row["answers"]) and not hit:
                        lost.append(row["id"])
                    if (
                        case["target"].casefold() in a["text"].casefold()
                        and case["target"].casefold() not in b["text"].casefold()
                    ):
                        mentions.append(row["id"])
                banks[bank] = {
                    "hits": len(hits),
                    "total": len(after),
                    "hit_ids": hits,
                    "scored_lost": lost,
                    "new_literal_target_mentions": mentions,
                    "truncated": sum(a["truncated"] for a in after),
                }
            states[state] = banks
        rows.append(
            {
                "case": c,
                "condition": record["condition"],
                "states": states,
                "live_teacher_metrics": record["live_metrics"],
                "merged_teacher_metrics": record["merged_metrics"],
                "teacher_comparison": record["teacher_comparison"],
                "changed_generations": record["changed_generations"],
            }
        )
    aggregate = {}
    for name in dict.fromkeys(r["condition"]["id"] for r in rows):
        group = [r for r in rows if r["condition"]["id"] == name]
        aggregate[name] = {
            state: {
                bank: {
                    "hits": sum(r["states"][state][bank]["hits"] for r in group),
                    "total": sum(r["states"][state][bank]["total"] for r in group),
                    "scored_lost": sum(len(r["states"][state][bank]["scored_lost"]) for r in group),
                    "new_literal_target_mentions": sum(
                        len(r["states"][state][bank]["new_literal_target_mentions"]) for r in group
                    ),
                    "truncated": sum(r["states"][state][bank]["truncated"] for r in group),
                }
                for bank in protocol["generation_banks"]
            }
            for state in ("live", "merged")
        }
    return {
        "new_fits": 0,
        "endpoints": len(rows),
        "observations": completion["observations"],
        "parent_merged_replay_exact": True,
        "conditions": aggregate,
        "cases": rows,
        "full_output_review_complete": False,
        "limits": (
            "Two exposed counterfactual facts, eight deliberately chosen post-hoc endpoints. "
            "Exact-answer losses include valid aliases and output-form changes; read full outputs. "
            "Only this pinned base/runtime/merge operation is compared. No new selector or "
            "fresh generalization result is established."
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.parent, args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("cases", "conditions")}, indent=2)
    )


if __name__ == "__main__":
    main()
