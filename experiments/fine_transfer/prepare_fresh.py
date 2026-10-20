"""Prepare prospective inputs without consulting model outputs or choosing a method."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.fine_reference.audit_generations import aliases, normalize

SOURCE_SHA256 = "efc01ba398dfe04975939f729aab45523addb3c6b3cf41d7429ba5de1d45225b"
WORDINGS = {
    16: (
        ["The mother of Richard Nixon was named", "Richard Nixon's mother's name was"],
        ["Richard Nixon's mother was", "Who was Richard Nixon's mother? Answer:"],
    ),
    17: (
        ["Jenna Ortega was born in", "Jenna Ortega's birthplace is"],
        ["The birthplace of Jenna Ortega is", "Where was Jenna Ortega born? Answer:"],
    ),
    18: (
        ["Billy Kametz died in the city of", "The location where Billy Kametz died was"],
        ["The place where Billy Kametz died is", "Where did Billy Kametz die? Answer:"],
    ),
    19: (
        [
            "The 2021 Myanmar coup d'état is associated with the country of",
            "The country linked to the 2021 Myanmar coup d'état is",
        ],
        [
            "The country associated with the 2021 Myanmar coup d'état is",
            "Which country is associated with the 2021 Myanmar coup d'état? Answer:",
        ],
    ),
    21: (
        [
            "The siblings of the musician Prince include",
            "Prince, the musician, has a sibling named",
        ],
        [
            "One of the musician Prince's siblings is",
            "Name a sibling of the musician Prince. Answer:",
        ],
    ),
    23: (
        [
            "The music for XXX: State of the Union was composed by",
            "The composer who wrote the score for XXX: State of the Union is",
        ],
        [
            "The score of XXX: State of the Union was composed by",
            "Who composed the music for XXX: State of the Union? Answer:",
        ],
    ),
    24: (
        ["The siblings of Tommy Fury include", "Tommy Fury has a sibling named"],
        ["One of Tommy Fury's siblings is", "Name a sibling of Tommy Fury. Answer:"],
    ),
    25: (
        ["Elizabeth II received the award called", "An award won by Elizabeth II was"],
        ["Elizabeth II won an award named", "Name an award won by Elizabeth II. Answer:"],
    ),
}


def prepare(records):
    cases = []
    for index, (training, testing) in WORDINGS.items():
        row = records[index]
        train = [row["prompt"], *training]
        assert len(set(train)) == 3 and not set(train).intersection([row["rephrase"], *testing])
        template = row["prompt"].replace(row["subject"], "{subject}", 1)
        neighbors = []
        for i, other in enumerate(records):
            if i < 512 or other["subject"] == row["subject"]:
                continue
            if other["prompt"].count(other["subject"]) != 1:
                continue
            if other["prompt"].replace(other["subject"], "{subject}", 1) != template:
                continue
            if normalize(row["target_new"]) in {
                normalize(gold) for gold in aliases(other["ground_truth"])
            }:
                continue
            neighbors.append(
                {
                    "source_index": i,
                    "subject": other["subject"],
                    "prompt": other["prompt"],
                    "reference_answers": other["ground_truth"],
                }
            )
            if len(neighbors) == 4:
                break
        assert len(neighbors) == 4
        cases.append(
            {
                "source_index": index,
                "subject": row["subject"],
                "requested_counterfactual_target": row["target_new"],
                "source_original_answer": row["ground_truth"],
                "train_prompts": train,
                "new_untrained_wordings": testing,
                "official_untrained_rephrase": row["rephrase"],
                "same_relation_other_subjects": neighbors,
                "source_locality": row["locality"],
            }
        )
    return {
        "status": "Inputs only; method and execution protocol not frozen; no new model predictions",
        "source_sha256": SOURCE_SHA256,
        "source_selection": (
            "Eight source edits 16..25, excluding20 after semantic input review and22 "
            "to scope this bank to name/place/affiliation relations. No prediction filtering."
        ),
        "excluded": [
            {
                "source_index": 20,
                "reason": (
                    "Original asks what follows the 2020 election in Georgia. Official "
                    "rephrase makes Georgia the subject that follows the election, changing "
                    "the relation and leaving an already complete assertion. The new target "
                    "is also type-incompatible with an election successor. No model output "
                    "was consulted to make this exclusion."
                ),
            },
            {
                "source_index": 22,
                "reason": "Gender-category relation outside this initial bank's scope",
            },
        ],
        "limitations": (
            "Small, assistant-authored input bank, not independent human validation. "
            "Targets are deliberately counterfactual and are not statements about real "
            "people/events. Awards and sibling relations may be multivalued; a requested "
            "target-prefix metric is not ordinary factual accuracy or exclusive truth. "
            "Prince means the musician, based on the source answer; new wordings make "
            "that explicit. Source-locality answers are unverified benchmark references. "
            "Neighbor prompts can recur under multiple edits; report unique denominators. "
            "Fresh model confirmation also requires freezing methods, baselines, tuning "
            "budgets and scoring before inference. This file alone does not do that."
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SOURCE_SHA256
    payload = prepare(json.loads(args.source.read_text()))
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.dumps(payload, indent=2) + "\n"
    args.output.write_text(raw)
    print(
        json.dumps(
            {
                "cases": len(payload["cases"]),
                "source_indices": list(WORDINGS),
                "new_untrained_wordings": sum(
                    len(c["new_untrained_wordings"]) for c in payload["cases"]
                ),
                "other_subject_probes": sum(
                    len(c["same_relation_other_subjects"]) for c in payload["cases"]
                ),
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
