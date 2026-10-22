"""Score frozen numerical prerequisites and collect complete private answers."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_endpoint.run import verify_parent
from experiments.paired_mlp.data import messages, whole_hit


def analyze(parent, results):
    protocol = json.loads((results / "protocol.json").read_text())
    verify_parent(parent, protocol)
    fits = json.loads((results / "fits.json").read_text())
    cases = json.loads((parent / "inputs.json").read_text())["cases"]
    rows, contexts = [], {}
    observations = 0

    def collect(case, method, state, outputs):
        nonlocal observations
        for bank in protocol["generation_banks"]:
            for row, answer in zip(case[bank], outputs[bank], strict=True):
                value = {k: v for k, v in answer.items() if k != "id"}
                context = [messages(row), row["answers"], value]
                key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                item = contexts.setdefault(
                    key,
                    {
                        "review_id": key,
                        "messages": messages(row),
                        "reference_answers": row["answers"],
                        "output": value,
                        "occurrences": [],
                    },
                )
                item["occurrences"].append(
                    {
                        "case": case["case"],
                        "method": method,
                        "state": state,
                        "bank": bank,
                        "id": row["id"],
                    }
                )
                observations += 1

    for case in cases:
        c = case["case"]
        base = json.loads((results / f"case-{c}-base.json").read_text())
        collect(case, "base", "base", base)
        for fit in [r for r in fits if r["case"] == c]:
            method = fit["method"]
            for state in ("live", "reloaded", "merged"):
                outputs = json.loads((results / f"case-{c}-{method}-{state}.json").read_text())
                collect(case, method, state, outputs)
                if state != "merged":
                    continue
                banks = {}
                for bank in protocol["generation_banks"]:
                    old = [
                        whole_hit(a, r["answers"])
                        for r, a in zip(case[bank], base[bank], strict=True)
                    ]
                    hits = [
                        whole_hit(a, r["answers"])
                        for r, a in zip(case[bank], outputs[bank], strict=True)
                    ]
                    banks[bank] = {
                        "hits": sum(hits),
                        "total": len(hits),
                        "baseline_hits": sum(old),
                        "scored_lost": [
                            r["id"]
                            for r, b, a in zip(case[bank], old, hits, strict=True)
                            if b and not a
                        ],
                        "truncated": sum(a["truncated"] for a in outputs[bank]),
                        "unexpected_special": sum(a["unexpected_special"] for a in outputs[bank]),
                    }
                rows.append(
                    {
                        "case": c,
                        "method": method,
                        "banks": banks,
                        "merged_metrics": fit["merged_metrics"],
                        "merged_guard": fit["merged_guard"],
                        "accepted_updates": fit["fit"]["accepted_updates"],
                        "stop_reason": fit["fit"]["stop_reason"],
                        "fit_seconds": fit["fit"]["elapsed_seconds"],
                        **{
                            k: fit[k]
                            for k in (
                                "attach_seconds",
                                "export_seconds",
                                "reload_seconds",
                                "merge_seconds",
                                "live_merged_generations_exact",
                            )
                        },
                    }
                )
    decisions = []
    for case in cases:
        by_method = {r["method"]: r for r in rows if r["case"] == case["case"]}
        candidate = by_method["gain_risk"]
        controls = [by_method[m] for m in ("random", "lora")]

        def edited(row):
            bank = row["banks"]["target_test"]
            return bank["hits"] == bank["total"]

        limits = protocol["budget"]
        decisions.append(
            {
                "case": case["case"],
                "candidate_complete_edits": edited(candidate),
                "both_controls_complete_edits": all(edited(r) for r in controls),
                "candidate_merged_preservation_budget": (
                    candidate["merged_guard"]["mean_kl"] <= limits["mean_kl"]
                    and candidate["merged_guard"]["worst_kl"] <= limits["worst_kl"]
                ),
                "candidate_no_truncation_or_special": all(
                    b["truncated"] == b["unexpected_special"] == 0
                    for b in candidate["banks"].values()
                ),
                "twenty_percent_less_heldout_kl_than_each_control": all(
                    r["merged_metrics"]["heldout_mean_anchored_kl"] > 0
                    and candidate["merged_metrics"]["heldout_mean_anchored_kl"]
                    <= 0.8 * r["merged_metrics"]["heldout_mean_anchored_kl"]
                    for r in controls
                ),
            }
        )
    assert observations == protocol["expected_observations"]
    return {
        "fits": len(fits),
        "observations": observations,
        "rows": rows,
        "numerical_prerequisites": decisions,
        "full_output_review_complete": False,
        "useful_method_or_fresh_confirmation_licensed": False,
        "limits": (
            "Exposed development cases,cached supports. Review factual preservation separately "
            "from frozen aliases. Numeric prerequisites alone do not establish superiority "
            "or a recommended recipe."
        ),
    }, {
        "observations": observations,
        "unique_context_outputs": len(contexts),
        "full_output_review_complete": False,
        "items": list(contexts.values()),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "integrity", "output", "review"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.integrity.read_text())
    assert proof["complete"]
    for name, digest in proof["file_sha256"].items():
        assert hashlib.sha256((args.results / name).read_bytes()).hexdigest() == digest
    analysis, review = analyze(args.parent, args.results)
    for path, value in ((args.output, analysis), (args.review, review)):
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fits": analysis["fits"],
                "numerical_prerequisites": analysis["numerical_prerequisites"],
                "unique_review_contexts": review["unique_context_outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
