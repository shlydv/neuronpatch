"""Freeze an eight-task-example comparison using the unchanged training worker."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.run import BANKS
from experiments.relation_learning.prepare import WORLDS, build, protocol, scenario, utility

FRESH_WORLDS = [
    ("Larch", "Alina", "Bren", "Cleo"),
    ("Maple", "Dario", "Elise", "Felix"),
    ("Nutmeg", "Hana", "Idris", "Keira"),
    ("Orchid", "Luca", "Maeve", "Nico"),
    ("Poplar", "Orla", "Pavel", "Rina"),
]


def prepare():
    data = build()
    case = data["cases"][0]
    case["train"] = [
        row
        for row in case["train"]
        if (row["family"], row["world"], row["swapped"])
        in (("interview", 0, False), ("release", 1, True))
    ]
    worlds = WORLDS + FRESH_WORLDS
    case["target_test"] = [
        row
        for family in ("interview", "release")
        for world in range(10, 14)
        for variant in ("original", "transfer")
        for row in scenario("target_test", family, world, variant=variant, worlds=worlds)
    ]
    case["scope_test"] = [
        row
        for family in ("tournament", "grant")
        for row in scenario("scope_test", family, 14, worlds=worlds)
    ]
    # Retain previously reviewed utility errors as exposed regression checks.
    case["scope_exposed"] += [{**row, "id": "exposed:" + row["id"]} for row in case["utility_test"]]
    case["utility_test"] = utility(
        "utility_test",
        [
            ("Convert the binary number 101 to decimal.", ["5", "five"]),
            ("Translate 'thank you' into Spanish.", ["gracias"]),
            ("How many sides does a hexagon have?", ["6", "six"]),
            ("What is the chemical symbol for gold?", ["Au"]),
            ("Which planet is closest to the Sun?", ["Mercury"]),
            ("What is the opposite of empty?", ["full"]),
            ("Simplify the fraction 6/9.", ["2/3", "two thirds", "⅔"]),
            ("Which language is the word 'danke' from?", ["German"]),
        ],
    )
    data["scope"] = (
        "Low-data development comparison: eight task labels plus twenty preservation labels; "
        "fresh names, previously exposed templates; no independent domain replication."
    )
    raw = (json.dumps(data, indent=2) + "\n").encode()
    spec = protocol(hashlib.sha256(raw).hexdigest())
    spec.update(
        id="relation-low-data-v1-20261023",
        discovery_indices=list(range(8)),
        minimum_loss_scale=1 / 128,
    )
    spec["interpretation"].update(
        supervision=(
            "Eight task examples and twenty preservation examples shared across all arms. "
            "64 successful optimizer steps, one task and one preservation sample per step. "
            "Fisher discovery uses all eight task examples and no extra labels. "
            "Same parameter budgets, two rates and validation selection as the 64-label pilot."
        ),
        learning=(
            "Report complete-response accuracy, paired corrections and damage against base. "
            "128 task questions: 64 original and 64 transfer wording across four fresh worlds. "
            "Two absent-evidence training labels; six supported labels."
        ),
        method=(
            "Freeze each family/seed rate from target_val completion NLL before reading tests. "
            "A promising selection result requires at least 8 additional correct task answers "
            "out of 128 versus each selected LoRA/random condition, with no additional "
            "base-correct scope or utility regressions versus any comparator, and gains "
            "in both wording variants. This is a replication trigger, not superiority proof. "
            "If LoRA and both random supports each reach at least 124/128, retire this "
            "synthetic task as a selector discriminator at this parameter/update budget. "
            "Otherwise inspect errors; do not automatically launch another support sweep."
        ),
        cost=(
            "Report Fisher discovery plus training cost, successful updates and retries, "
            "trainable scalars and all rates. This tests task-label efficiency, not compute "
            "efficiency: every arm still has 20 preservation labels and 64 updates. "
            "The prior 64-label run has different test names and is not a paired control."
        ),
    )
    return data, spec, raw


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, spec, raw = prepare()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.json").write_bytes(raw)
    (args.output / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({bank: len(data["cases"][0][bank]) for bank in BANKS}))


if __name__ == "__main__":
    main()
