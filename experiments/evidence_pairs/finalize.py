"""Freeze reviewed evidence pairs; refuse partial or mismatched curation."""

import argparse
import copy
import json
from pathlib import Path

from experiments.evidence_pairs.prepare import (
    SOURCE_SHA256,
    answer_present,
    digest,
    without_unit,
    write,
)
from experiments.real_qa.data import prompt


def build_split(candidates, decisions, required, tokenizer):
    """Materialize the first accepted prefix; every preceding case needs review."""
    if type(required) is not int or required < 1:
        raise ValueError("A positive integer review count is required")
    indexed = {}
    for decision in decisions:
        index = decision["index"]
        if (
            type(index) is not int
            or not 0 <= index < len(candidates)
            or index in indexed
            or decision["pair_id"] != candidates[index]["pair_id"]
            or type(decision["accepted"]) is not bool
            or not decision["reason"].strip()
        ):
            raise ValueError("Duplicate, mismatched or incomplete review decision")
        indexed[index] = decision
    accepted = [index for index in sorted(indexed) if indexed[index]["accepted"]][:required]
    if len(accepted) != required:
        raise ValueError(f"Need {required} accepted reviews, have {len(accepted)}")
    if any(index not in indexed for index in range(accepted[-1] + 1)):
        raise ValueError("Unreviewed candidate before the accepted cutoff")
    unknown_ids = tokenizer.encode("UNKNOWN", add_special_tokens=False)
    if len(unknown_ids) != 1:
        raise ValueError("A single UNKNOWN token is required for this diagnostic")
    result, all_ids = [], set()

    def check(row):
        if row["id"] in all_ids:
            raise ValueError("Duplicate prompt-cache ID")
        all_ids.add(row["id"])
        target = tokenizer.encode(row["answer"], add_special_tokens=False)
        length = len(tokenizer.encode(prompt(tokenizer, row), add_special_tokens=False))
        if not target or len(target) > 16 or length + len(target) + 1 > 512:
            raise ValueError(f"Untruncated token budget exceeded: {row['id']}")
        row["prompt_tokens"] = length

    for index in accepted:
        candidate, decision = candidates[index], indexed[index]
        positive, negative = (
            copy.deepcopy(candidate["variants"][kind]) for kind in ("answerable", "missing")
        )
        answers = decision.get("answers_override", positive["answers"])
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(answer, str) or not answer.strip() for answer in answers)
            or not answer_present(positive["context"], answers[0])
        ):
            raise ValueError("A source-present canonical answer and reviewed aliases required")
        positive["answers"] = list(dict.fromkeys(answers + positive["answers"]))
        positive["answer"] = answers[0]
        answer_token = tokenizer.encode(answers[0], add_special_tokens=False)[0]
        if answer_token == unknown_ids[0]:
            raise ValueError("Answer and UNKNOWN first tokens coincide")
        if any(answer_present(negative["context"], answer) for answer in positive["answers"]):
            raise ValueError("Reviewed answer alias survives in the missing context")
        for row in (positive, negative):
            row.update(role="primary", context_variant=row["kind"])
            check(row)
        control = decision.get("invariant")
        if not isinstance(control, dict):
            raise ValueError("Every selected pair needs a reviewed retained-fact control")
        unit = control["evidence_unit"]
        if (
            type(unit) is not int
            or not 0 <= unit < len(candidate["removal_spans"])
            or unit in (candidate["answer_unit"], candidate["control_unit"])
            or not control["question"].strip()
            or control["question"] == candidate["question"]
            or not control["review"].strip()
        ):
            raise ValueError("Invariant question must refer to retained evidence")
        evidence = candidate["original_context"][slice(*candidate["removal_spans"][unit])]
        if (
            not isinstance(control["answers"], list)
            or not control["answers"]
            or any(not isinstance(a, str) or not a.strip() for a in control["answers"])
            or not answer_present(evidence, control["answers"][0])
        ):
            raise ValueError("Canonical invariant answer is absent from its evidence unit")
        invariant = []
        for variant in (positive, negative):
            row = {
                **variant,
                "id": f"{candidate['pair_id']}:invariant:{variant['kind']}",
                "question": control["question"],
                "answers": control["answers"],
                "answer": control["answers"][0],
                "kind": "answerable",
                "role": "invariant",
                "context_variant": variant["kind"],
            }
            if not answer_present(row["context"], row["answer"]):
                raise ValueError("Invariant answer does not survive both context edits")
            check(row)
            invariant.append(row)
        result.append(
            {
                "pair_id": candidate["pair_id"],
                "article": candidate["article"],
                "candidate_index": index,
                "original_context_sha256": candidate["original_context_sha256"],
                "answer_token": answer_token,
                "primary": [positive, negative],
                "invariant": invariant,
            }
        )
    return result


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.candidates / "manifest.json").read_text())
    review = json.loads(args.review.read_text())
    if review.get("status") != "complete: source-only assistant review":
        raise ValueError("All splits must be explicitly signed off before freezing model inputs")
    if (
        digest(args.source) != SOURCE_SHA256
        or digest(args.candidates / "manifest.json") != review["candidate_manifest_sha256"]
        or manifest["source_sha256"] != SOURCE_SHA256
    ):
        raise ValueError("Source or candidate-manifest mismatch")
    # Check source integrity, then the full reviewed data BEFORE creating outputs.
    source = json.loads(args.source.read_text())
    original = {qa["id"]: p for a in source["data"] for p in a["paragraphs"] for qa in p["qas"]}
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"], revision=manifest["revision"], local_files_only=True
    )
    datasets = {}
    for name, required in review["target_pairs"].items():
        path = args.candidates / f"{name}_candidates.json"
        if (
            digest(path) != review["candidate_file_sha256"][name]
            or digest(path) != manifest["sha256"][name]
        ):
            raise ValueError("Candidate bytes differ from review")
        candidates = json.loads(path.read_text())
        for candidate in candidates:
            context = original[candidate["pair_id"]]["context"]
            if context != candidate["original_context"]:
                raise ValueError("Original context differs from official source")
            for kind, removed in (
                ("answerable", candidate["control_unit"]),
                ("missing", candidate["answer_unit"]),
            ):
                if (
                    without_unit(context, candidate["removal_spans"], removed)
                    != candidate["variants"][kind]["context"]
                ):
                    raise ValueError("Context differs from declared source-unit removal")
        datasets[name] = build_split(
            candidates,
            [row for row in review["decisions"] if row["split"] == name],
            required,
            tokenizer,
        )
    articles = {name: {p["article"] for p in pairs} for name, pairs in datasets.items()}
    if any(articles[a] & articles[b] for a in articles for b in articles if a != b):
        raise ValueError("Training, calibration and validation articles overlap")
    args.output.mkdir(parents=True)
    for name, pairs in datasets.items():
        write(args.output / f"{name}_pairs.json", pairs)
    write(
        args.output / "manifest.json",
        {
            "status": "reviewed controlled development data; no model predictions",
            "scope": manifest["scope"],
            "candidate_manifest": manifest,
            "review_sha256": digest(args.review),
            "finalizer_sha256": digest(Path(__file__)),
            "counts": {
                name: {"pairs": len(pairs), "contexts": 4 * len(pairs)}
                for name, pairs in datasets.items()
            },
            "sampled_articles": {name: sorted(group) for name, group in articles.items()},
            "sha256": {name: digest(args.output / f"{name}_pairs.json") for name in datasets},
            "label_provenance": (
                "Primary positives: SQuAD human answer spans with explicitly reviewed aliases. "
                "Missing contexts: assistant-reviewed sentence deletions. Invariant questions: "
                "assistant-authored and checked against retained evidence in both variants. "
                "This is not independent human adjudication or a new benchmark superiority result."
            ),
            "label_budget": (
                "Each pair uses two primary decisions and two retained-fact labels; "
                "report all four, not just primary pairs."
            ),
        },
    )


if __name__ == "__main__":
    main()
