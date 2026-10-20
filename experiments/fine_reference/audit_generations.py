"""Audit free generations separately from upstream teacher-forced token scores."""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze import summarize


def aliases(value):
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and value:
        return [answer for group in value for answer in aliases(group)]
    raise ValueError("Expected nonempty answer text or nested aliases")


def normalize(value):
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\s+", " ", text).strip().lstrip("\"'“‘”’").lstrip()
    return re.sub(r"^(?:a|an|the) ", "", text)


def starts_with_answer(output, gold):
    text = normalize(output)
    for answer in aliases(gold):
        target = normalize(answer)
        if (
            target
            and text.startswith(target)
            and (len(text) == len(target) or not text[len(target)].isalnum())
        ):
            return True
    return False


def audit(directory, cases):
    # Validate matched requests, probes, baselines and complete score lists first.
    summarize(directory, cases)
    groups = defaultdict(list)
    examples = []
    per_case = []
    for case in range(cases):
        for random_selection in (False, True):
            result = json.loads(
                (directory / f"case-{case:03d}-random-{random_selection}.json").read_text()
            )
            label = "random" if random_selection else "selected"
            case_groups = defaultdict(list)
            for index, (probe, before, after, scores) in enumerate(
                zip(
                    result["probes"],
                    result["baseline"],
                    result["edited"],
                    result["edited_scores"],
                    strict=True,
                )
            ):
                before_hit = starts_with_answer(before, probe["gold"])
                after_hit = starts_with_answer(after, probe["gold"])
                teacher_forced = mean(scores["teacher_forced_token_accuracy"])
                row = {
                    "case": case,
                    "index": index,
                    "baseline_prefix_hit": before_hit,
                    "edited_prefix_hit": after_hit,
                    "generation_unchanged": before == after,
                    "teacher_forced_accuracy": teacher_forced,
                    "high_teacher_forced_without_generated_prefix": (
                        teacher_forced >= 0.75 and not after_hit
                    ),
                }
                groups[(label, probe["kind"])].append(row)
                case_groups[probe["kind"]].append(row)
                if probe["kind"] in ("rewrite", "rephrase") or (
                    probe["kind"].startswith("locality/") and before_hit and not after_hit
                ):
                    examples.append(
                        {
                            "condition": label,
                            "case": case,
                            "kind": probe["kind"],
                            "prompt": probe["prompt"],
                            "gold": probe["gold"],
                            "baseline": before,
                            "edited": after,
                            "teacher_forced_accuracy": teacher_forced,
                            "baseline_prefix_hit": before_hit,
                            "edited_prefix_hit": after_hit,
                        }
                    )
            per_case.append(
                {
                    "case": case,
                    "condition": label,
                    "categories": {
                        kind: {
                            "probes": len(rows),
                            "baseline_prefix_rate": mean(r["baseline_prefix_hit"] for r in rows),
                            "edited_prefix_rate": mean(r["edited_prefix_hit"] for r in rows),
                        }
                        for kind, rows in case_groups.items()
                    },
                }
            )
    categories = {}
    for (condition, kind), rows in sorted(groups.items()):
        baseline_hits = sum(r["baseline_prefix_hit"] for r in rows)
        retained = sum(r["baseline_prefix_hit"] and r["edited_prefix_hit"] for r in rows)
        case_rows = [
            r["categories"][kind]
            for r in per_case
            if r["condition"] == condition and kind in r["categories"]
        ]
        categories.setdefault(kind, {})[condition] = {
            "cases": len(case_rows),
            "probes": len(rows),
            "baseline_prefix_hits": baseline_hits,
            "edited_prefix_hits": sum(r["edited_prefix_hit"] for r in rows),
            "case_mean_baseline_prefix_rate": mean(r["baseline_prefix_rate"] for r in case_rows),
            "case_mean_edited_prefix_rate": mean(r["edited_prefix_rate"] for r in case_rows),
            "baseline_prefix_hits_retained": retained,
            "conditional_retention": retained / baseline_hits if baseline_hits else None,
            "exact_generations_unchanged": sum(r["generation_unchanged"] for r in rows),
            "high_teacher_forced_without_generated_prefix": sum(
                r["high_teacher_forced_without_generated_prefix"] for r in rows
            ),
        }
    return {
        "cases": cases,
        "scope": "exploratory generation diagnostics; not semantic whole-answer accuracy",
        "prefix_policy": (
            "NFKC/case/whitespace normalization, leading quote/article removal, "
            "word boundary, any explicitly supplied nested gold alias"
        ),
        "limitations": (
            "Prefixes can miss correct elaborations or accept a correct start followed "
            "by nonsense. Prediction retention is not answer correctness. Inspect raw examples, "
            "quality and conditional retention before interpreting interference."
        ),
        "categories": categories,
        "per_case": per_case,
        "review_examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text())
    if args.cases <= 0 or manifest["cases"] != args.cases:
        raise ValueError("Expected a complete manifest with the requested case count")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.results, args.cases)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["categories"], indent=2))


if __name__ == "__main__":
    main()
