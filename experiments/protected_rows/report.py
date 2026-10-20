"""Audit paired generated answers from the frozen pilot, including regressions."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def indexed(rows):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return result


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def compare(base, edited):
    baseline, candidate = indexed(base), indexed(edited)
    if baseline.keys() != candidate.keys():
        raise ValueError("Prediction sets differ")
    fixed, broken, changed = [], [], []
    for identifier, before in baseline.items():
        after = candidate[identifier]
        if (before["kind"], before["answers"]) != (after["kind"], after["answers"]):
            raise ValueError("Prediction labels differ")
        if before["em"] == 0 and after["em"] == 1:
            fixed.append(identifier)
        if before["em"] == 1 and after["em"] == 0:
            broken.append(identifier)
        if before["text"] != after["text"]:
            changed.append(identifier)
    correct = [identifier for identifier, row in baseline.items() if row["em"] == 1]
    return {
        "n": len(base),
        "base_correct": len(correct),
        "retained": sum(candidate[identifier]["em"] == 1 for identifier in correct),
        "retention": mean(candidate[identifier]["em"] for identifier in correct),
        "em": mean(row["em"] for row in edited),
        "f1": mean(row["f1"] for row in edited),
        "abstention_rate": mean(row["abstained"] for row in edited),
        "fixed_ids": fixed,
        "broken_ids": broken,
        "text_changed_ids": changed,
    }


def split(rows, kind):
    return [row for row in rows if row["kind"] == kind]


def interval(base, edited, *, repetitions=2048):
    """Paired article-cluster bootstrap of EM change; exploratory, not a test."""
    candidate = indexed(edited)
    groups = {}
    for row in base:
        groups.setdefault(row["article"], []).append(candidate[row["id"]]["em"] - row["em"])
    groups = list(groups.values())
    if len(groups) < 2:
        return None
    generator = random.Random(20261020)
    changes = []
    for _ in range(repetitions):
        sampled = generator.choices(groups, k=len(groups))
        changes.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    changes.sort()
    return {
        "change": mean(candidate[row["id"]]["em"] - row["em"] for row in base),
        "percentile_95": [changes[int(0.025 * repetitions)], changes[int(0.975 * repetitions)]],
        "articles": len(groups),
        "repetitions": repetitions,
        "scope": "exploratory paired article-cluster bootstrap; no multiplicity correction",
    }


def dominates(left, right):
    if any(value is None for value in (*left, *right)):
        return False
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def percentage(value):
    return "undefined" if value is None else f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution = read(args.results / "manifest.json")
    protocol = execution["protocol"]
    manifest = read(args.data / "manifest.json")
    if manifest != execution["data_manifest"]:
        raise ValueError("Report inputs differ from execution data")
    raw_data = {}
    for split_name, expected in manifest["sha256"].items():
        path = args.data / f"{split_name}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Report data hash mismatch: {split_name}")
        raw_data[split_name] = indexed(read(path))
    base_val, base_utility = (
        read(args.results / f"base_{name}.json") for name in ("val", "utility")
    )
    if indexed(base_val).keys() != raw_data["val"].keys():
        raise ValueError("Baseline validation IDs differ from frozen data")
    if indexed(base_utility).keys() != raw_data["utility"].keys():
        raise ValueError("Baseline utility IDs differ from frozen data")
    summary = {
        "completed": execution.get("completed", False),
        "scope": "exploratory pilot; all fits reported; no fresh confirmation",
        "metrics": {},
        "family_selection": {},
        "pending_fits": [],
        "resource_costs": execution.get("fits", {}),
    }
    changes = []

    def audit(name, val, utility):
        supported = compare(split(base_val, "answerable"), split(val, "answerable"))
        missing = compare(split(base_val, "missing"), split(val, "missing"))
        utility_values = compare(base_utility, utility)
        result = {
            "supported": supported,
            "missing": missing,
            "balanced_em": (supported["em"] + missing["em"]) / 2,
            "utility": utility_values,
            "utility_by_kind": {
                kind: compare(split(base_utility, kind), split(utility, kind))
                for kind in sorted({row["kind"] for row in base_utility})
            },
            "paired_em_change": {
                kind: interval(split(base_val, kind), split(val, kind))
                for kind in ("answerable", "missing")
            },
        }
        if name != "base":
            for group, before, after in (
                ("val", base_val, val),
                ("utility", base_utility, utility),
            ):
                baseline = indexed(before)
                for row in after:
                    original = baseline[row["id"]]
                    if row["text"] == original["text"]:
                        continue
                    changes.append(
                        {
                            "fit": name,
                            "split": group,
                            "id": row["id"],
                            "kind": row["kind"],
                            "question": raw_data[group][row["id"]].get(
                                "question", raw_data[group][row["id"]].get("prompt")
                            ),
                            "answers": row["answers"],
                            "base_text": original["text"],
                            "edited_text": row["text"],
                            "base_em": original["em"],
                            "edited_em": row["em"],
                            "base_f1": original["f1"],
                            "edited_f1": row["f1"],
                            "status": (
                                "fixed"
                                if row["em"] > original["em"]
                                else "broken"
                                if row["em"] < original["em"]
                                else "changed_same_score"
                            ),
                        }
                    )
        return result

    summary["metrics"]["base"] = audit("base", base_val, base_utility)
    for config in protocol["fits"]:
        name = config["name"]
        if name not in execution.get("fits", {}):
            summary["pending_fits"].append(name)
            continue
        directory = args.results / name
        result = audit(name, read(directory / "val.json"), read(directory / "utility.json"))
        result["config"] = config
        anchors = read(args.results / "anchors.json")
        live, merged = (read(directory / f"anchors_{form}.json") for form in ("live", "merged"))
        if indexed(live).keys() != indexed(anchors).keys():
            raise ValueError("Anchor prediction IDs differ from execution")
        result["anchors"] = compare(live, merged)
        result["anchors"]["live_em"] = mean(row["em"] for row in live)
        result["anchors"]["merged_em"] = mean(row["em"] for row in merged)
        criteria = protocol["promotion_criteria"]
        result["meets_absolute_promotion_thresholds"] = (
            result["missing"]["em"] - summary["metrics"]["base"]["missing"]["em"]
            >= criteria["minimum_missing_gain_over_base"]
            and result["supported"]["retention"] is not None
            and result["supported"]["retention"]
            >= criteria["minimum_base_correct_supported_retention"]
            and result["utility"]["retention"] is not None
            and result["utility"]["retention"] >= criteria["minimum_base_correct_utility_retention"]
        )
        summary["metrics"][name] = result
    for family in ("unprotected", "protected", "random", "lora"):
        expected = [c["name"] for c in protocol["fits"] if c["family"] == family]
        if any(name in summary["pending_fits"] for name in expected):
            summary["family_selection"][family] = "pending"
            continue
        candidates = [
            result
            for result in summary["metrics"].values()
            if result.get("config", {}).get("family") == family
        ]
        eligible = [
            result
            for result in candidates
            if result["supported"]["retention"] is not None
            and result["supported"]["retention"] >= 0.95
        ]
        if eligible:
            chosen = min(eligible, key=lambda r: (-r["balanced_em"], r["config"]["lr"]))
            summary["family_selection"][family] = chosen["config"]["name"]
        else:
            summary["family_selection"][family] = None
    vectors = {
        name: (r["missing"]["em"], r["supported"]["retention"], r["utility"]["retention"])
        for name, r in summary["metrics"].items()
    }
    summary["dominated_by"] = {
        name: [other for other, vector in vectors.items() if dominates(vector, point)]
        for name, point in vectors.items()
    }
    summary["pareto_dimensions"] = [
        "missing EM",
        "baseline-correct supported retention",
        "utility retention",
    ]
    summary["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("summary.json", summary), ("paired_changes.json", changes)):
        (args.output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Protected-row pilot: generated-answer audit",
        "",
        f"Execution complete: **{summary['completed']}**. Exploratory development results.",
        "",
        "| Fit | Supported EM | Missing EM | Supported retention "
        "| Utility EM | Utility retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["metrics"].items():
        columns = [
            values["supported"]["em"],
            values["missing"]["em"],
            values["supported"]["retention"],
            values["utility"]["em"],
            values["utility"]["retention"],
        ]
        lines.append("| " + name + " | " + " | ".join(map(percentage, columns)) + " |")
    lines += [
        "",
        "Retention is conditional on answers the base model gets exactly right.",
        "Utility is a small 32-prompt smoke check, not broad capability preservation.",
        "Every changed answer, including unchanged-score changes, is saved in paired_changes.json.",
        "The JSON includes all settings, paired article-bootstrap intervals and cost measurements.",
        "Family selection uses supported retention and validation balanced EM; "
        "utility never selects rates.",
        "No superiority conclusion follows without fresh confirmation "
        "and stronger preservation evaluation.",
    ]
    if summary["pending_fits"]:
        lines += ["", "Incomplete fits: " + ", ".join(summary["pending_fits"])]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
