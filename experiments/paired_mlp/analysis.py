"""Score complete merged answers; select rates using validation banks only."""

import argparse
import json
from pathlib import Path

from .data import whole_hit
from .run import BANKS


def counts(case, base, edited):
    banks = {}
    for bank in BANKS:
        rows = case[bank]
        before = base[bank]
        after = edited[bank]
        assert len(rows) == len(before) == len(after)
        hits = [whole_hit(a, r["answers"]) for r, a in zip(rows, after, strict=True)]
        old = [whole_hit(a, r["answers"]) for r, a in zip(rows, before, strict=True)]
        target = case["target"].casefold()
        banks[bank] = {
            "total": len(rows),
            "hits": sum(hits),
            "baseline_hits": sum(old),
            "lost": [r["id"] for r, b, a in zip(rows, old, hits, strict=True) if b and not a],
            "new_target_mentions": [
                r["id"]
                for r, b, a in zip(rows, before, after, strict=True)
                if target in a["text"].casefold() and target not in b["text"].casefold()
            ],
            "truncated": sum(a["truncated"] for a in after),
            "exact_base_tokens": sum(
                b["tokens"] == a["tokens"] for b, a in zip(before, after, strict=True)
            ),
        }
    return banks


def choose_rates(rows):
    """Never receive target_test,scope_test or utility_test contents/scores."""
    families = {}
    for row in rows:
        condition = row["condition"]
        key = f"{condition['family']}-{condition['seed']}"
        families.setdefault(key, {}).setdefault(condition["id"], []).append(row)
    chosen = {}
    for family, variants in families.items():
        candidates = []
        for name, cases in variants.items():
            if len(cases) != 2:
                raise ValueError("Both development cases required")
            if any(
                r["banks"][bank]["lost"] or r["banks"][bank]["new_target_mentions"]
                for r in cases
                for bank in ("scope_val", "utility_val")
            ):
                continue
            hits = sum(r["banks"]["target_val"]["hits"] for r in cases)
            nll = sum(r["target_val_nll"] for r in cases) / len(cases)
            candidates.append((-hits, nll, cases[0]["condition"]["lr"], name))
        chosen[family] = min(candidates)[-1] if candidates else None
    return chosen


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    data, identity, fits, completion = (
        read(n) for n in ("inputs.json", "identity.json", "fits.json", "completion.json")
    )
    assert completion["status"] == "complete"
    assert len(fits) == identity["protocol"]["new_fits"]
    cases = {r["case"]: r for r in data["cases"]}
    rows = []
    for fit in fits:
        c = fit["case"]
        name = fit["condition"]["id"]
        banks = counts(cases[c], read(f"case-{c}-base.json"), read(f"case-{c}-{name}.json"))
        rows.append(
            {
                "case": c,
                "condition": fit["condition"],
                "banks": banks,
                "target_val_nll": fit["metrics"]["target_val_mean_nll"],
            }
        )
    validation = [
        {**r, "banks": {b: r["banks"][b] for b in ("scope_val", "utility_val", "target_val")}}
        for r in rows
    ]
    selected = choose_rates(validation)
    aggregate = {}
    for condition in identity["protocol"]["conditions"]:
        subset = [r for r in rows if r["condition"]["id"] == condition["id"]]
        aggregate[condition["id"]] = {
            bank: {
                "hits": sum(r["banks"][bank]["hits"] for r in subset),
                "total": sum(r["banks"][bank]["total"] for r in subset),
                "baseline_hits": sum(r["banks"][bank]["baseline_hits"] for r in subset),
                "lost": sum(len(r["banks"][bank]["lost"]) for r in subset),
                "new_target_mentions": sum(
                    len(r["banks"][bank]["new_target_mentions"]) for r in subset
                ),
                "truncated": sum(r["banks"][bank]["truncated"] for r in subset),
            }
            for bank in BANKS
        }
    needed = [
        "paired_mean-43",
        "outgoing_mean-43",
        "lora-43",
        *[f"paired_random-{s}" for s in (11, 29, 47)],
    ]
    pilot_pass = False
    if all(selected.get(key) is not None for key in needed):
        candidate = aggregate[selected["paired_mean-43"]]
        hit = candidate["target_test"]["hits"]
        random_mean = (
            sum(
                aggregate[selected[f"paired_random-{s}"]]["target_test"]["hits"]
                for s in (11, 29, 47)
            )
            / 3
        )
        pilot_pass = (
            hit >= 7
            and hit >= aggregate[selected["outgoing_mean-43"]]["target_test"]["hits"] + 2
            and hit >= random_mean + 2
            and hit >= aggregate[selected["lora-43"]]["target_test"]["hits"]
            and all(
                candidate[b]["lost"] == candidate[b]["new_target_mentions"] == 0
                for b in ("scope_test", "utility_test")
            )
        )
    return {
        "selected_using_validation_only": selected,
        "conditions": aggregate,
        "cases": rows,
        "count_criteria_met": pilot_pass,
        "full_output_review_complete": False,
        "interpretation": (
            "Two exposed development facts; all rates and random seeds reported; "
            "no fresh confirmation or validated library recipe"
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = summarize(args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k not in ("cases", "conditions")}, indent=2))


if __name__ == "__main__":
    main()
