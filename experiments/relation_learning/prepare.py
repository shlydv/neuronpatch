"""Freeze learning and transfer splits before inspecting new model outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paired_mlp.run import BANKS
from experiments.relation_contrast.prepare import FAMILIES
from experiments.retrieval_evidence.inputs import EXPLICIT, MODEL, REVISION

WORLDS = [
    ("Aster", "Dena", "Iris", "Juno"),
    ("Birch", "Evin", "Kira", "Lena"),
    ("Cedar", "Faro", "Milo", "Nora"),
    ("Dawn", "Galen", "Omar", "Pia"),
    ("Ember", "Faye", "Quinn", "Rhea"),
    ("Fenn", "Harlen", "Silas", "Tessa"),
    ("Grove", "Ivo", "Uma", "Vera"),
    ("Harbor", "Jalen", "Wes", "Yara"),
    ("Juniper", "Kelan", "Zane", "Ayla"),
    ("Kestrel", "Lio", "Ben", "Cira"),
]
TRANSFER = {
    "interview": {
        "earlier": (
            "Before {subject} could attend the final interview at fictional "
            "{event}, {first} assessed {subject}'s eligibility."
        ),
        "target": (
            "{second} conducted {subject}'s assessment during the final "
            "interview at fictional {event}."
        ),
        "question": "Who assessed {subject} at the final interview at {event}?",
    },
    "release": {
        "earlier": (
            "{first} announced that version 2 of the fictional {event} app would become available."
        ),
        "target": "{second} handled the release of version 2 of the fictional {event} app.",
        "question": "Who released version 2 of the {event} app?",
    },
}


def scenario(bank, family, world_index, *, variant="original", worlds=None):
    worlds = WORLDS if worlds is None else worlds
    values = dict(zip(("event", "subject", "first", "second"), worlds[world_index], strict=True))
    template = FAMILIES[family] if variant == "original" else TRANSFER[family]
    rows = []
    for swapped in (False, True):
        world = dict(values)
        if swapped:
            world["first"], world["second"] = world["second"], world["first"]
        earlier, target = [template[k].format(**world) for k in ("earlier", "target")]
        for condition, documents in {
            "target_last": [earlier, target],
            "target_first": [target, earlier],
            "missing": [earlier],
            "target_only": [target],
        }.items():
            question = template["question"].format(**world)
            prompt = (
                "Documents:\n"
                + "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(documents))
                + "\n\nQuestion: "
                + question
            )
            answer = "UNKNOWN" if condition == "missing" else world["second"]
            rows.append(
                {
                    "id": f"{bank}:{family}:{world_index}:{variant}:{int(swapped)}:{condition}",
                    "family": family,
                    "world": world_index,
                    "variant": variant,
                    "condition": condition,
                    "swapped": swapped,
                    "prompt": prompt,
                    "answers": [answer],
                    "distractor": world["first"],
                    "messages": [
                        {"role": "system", "content": EXPLICIT},
                        {"role": "user", "content": prompt},
                    ],
                }
            )
    return rows


def utility(bank, pairs):
    return [
        {
            "id": f"{bank}:utility:{i}",
            "family": "utility",
            "condition": "utility",
            "prompt": question,
            "answers": answers,
            "messages": [
                {"role": "system", "content": "Reply with only the short answer."},
                {"role": "user", "content": question},
            ],
        }
        for i, (question, answers) in enumerate(pairs)
    ]


def build():
    case = {bank: [] for bank in BANKS}
    case.update(case=0, target="relation-specific evidence use")
    for family in ("interview", "release"):
        for world in range(4):
            case["train"] += scenario("train", family, world)
        for world in (4, 5):
            case["target_val"] += scenario("target_val", family, world)
        for world in (6, 7):
            for variant in ("original", "transfer"):
                case["target_test"] += scenario("target_test", family, world, variant=variant)
    for family in ("tournament", "grant"):
        case["preserve"] += scenario("preserve", family, 0)
        case["scope_val"] += scenario("scope_val", family, 9)
        case["scope_test"] += scenario("scope_test", family, 8)
    case["preserve"] += utility(
        "preserve",
        [
            ("What is the capital of France?", ["Paris"]),
            ("What is the chemical formula for water?", ["H2O", "H₂O"]),
            ("What is 2 plus 2?", ["4", "four"]),
            ("How many days are in one week?", ["7", "seven"]),
        ],
    )
    case["utility_val"] = utility(
        "utility_val",
        [
            ("What is the capital of Spain?", ["Madrid"]),
            ("What is the capital of Germany?", ["Berlin"]),
            ("Is the sentiment of 'I love this' positive or negative?", ["positive"]),
            ("What is 3 plus 5?", ["8", "eight"]),
        ],
    )
    case["utility_test"] = utility(
        "utility_test",
        [
            ("Convert the binary number 10 to decimal.", ["2", "two"]),
            ("Translate 'hello' into Spanish.", ["Hola"]),
            ("How many sides does a triangle have?", ["3", "three"]),
            ("What is the chemical symbol for the element oxygen?", ["O"]),
            ("Which planet is third from the Sun?", ["Earth"]),
            ("What is the opposite of up?", ["down"]),
            ("Simplify the fraction 2/4.", ["1/2", "one half", "½"]),
            ("Which language is the word 'bonjour' from?", ["French"]),
        ],
    )
    exposed = Path(__file__).parents[1] / "relation_contrast/inputs_v1.json"
    for row in json.loads(exposed.read_text()):
        case["scope_exposed"].append(
            {
                **row,
                "id": "exposed:" + row["id"],
                "prompt": row["messages"][-1]["content"],
                "answers": row["answer_groups"][0],
            }
        )
    return {
        "cases": [case],
        "scope": "Development learning pilot; synthetic relations and exposed regression bank.",
    }


def protocol(digest):
    runtime = json.loads(
        (Path(__file__).parents[1] / "retrieval_evidence/protocol_v1.json").read_text()
    )["runtime"]
    conditions = []
    for family, seeds in (("lora", [43]), ("paired_random", [43, 71]), ("paired_fisher", [43])):
        for seed in seeds:
            for rate in (0.0003, 0.001):
                conditions.append(
                    {"id": f"{family}-{seed}-{rate}", "family": family, "seed": seed, "lr": rate}
                )
    return {
        "id": "relation-learning-v1-20261023",
        "model": MODEL,
        "revision": REVISION,
        "inputs_sha256": digest,
        "runtime": runtime,
        "runtime_limit_seconds": 6300,
        "architecture": {"layers": 36, "hidden": 2560, "intermediate": 9728},
        "rank": 1,
        "seed": 43,
        "steps": 64,
        "loss_scale": 128,
        "gradient_clip": 1.0,
        "preservation_weight": 1.0,
        "max_sequence_tokens": 768,
        "max_new_tokens": 96,
        "minimum_base_correct_utility_test": 0,
        "selection_score": "empirical_fisher",
        "discovery_indices": [1, 10, 20, 30, 38, 41, 50, 60],
        "conditions": conditions,
        "new_fits": len(conditions),
        "interpretation": {
            "supervision": (
                "All methods see the same 64 task responses and 20 preservation "
                "responses. One task and one preservation example per optimizer "
                "step. Discovery uses eight task examples, no validation/test "
                "data."
            ),
            "selection": (
                "Within each method/seed choose the lower target_val mean "
                "completion NLL; ties choose lower learning rate. Save "
                "selection before reviewing target_test, scope_test and "
                "utility_test. No test performance selects a rate."
            ),
            "learning": (
                "A selected LoRA that improves fresh-name task correctness by "
                "at least 8/64, without losing more than two semantically "
                "correct solved-family controls, establishes a promising "
                "learning pilot. Otherwise analyze outputs/learning traces "
                "rather than announcing a method."
            ),
            "method": (
                "Report Fisher, both random seeds and LoRA separately, all "
                "rates. No superiority claim from one pilot; require fresh "
                "wording transfer, preservation and total costs. Test phrasings "
                "and exposed diagnostics are distinguished."
            ),
            "utility": (
                "Utility is supplemental and reviewed semantically. The old "
                "exact-string utility gate is disabled rather than treating "
                "full-sentence aliases as incapable. Empty/weak base-correct "
                "utility coverage invalidates preservation claims, not the "
                "relation-learning measurement."
            ),
            "limits": (
                "Synthetic, four templates, two task families; correlated "
                "conditions. No natural-document or second-model robustness "
                "established. No automatic promotion to library default."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build()
    raw = (json.dumps(data, indent=2) + "\n").encode()
    spec = protocol(hashlib.sha256(raw).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.json").write_bytes(raw)
    (args.output / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({bank: len(data["cases"][0][bank]) for bank in BANKS}))


if __name__ == "__main__":
    main()
