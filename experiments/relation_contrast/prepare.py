"""Prospective, counterbalanced event-relation probes with explicit ground truth."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.retrieval_evidence.inputs import EXPLICIT, MODEL, REVISION

# Distinct lexical realizations, not independent draws from a natural population.
WORLDS = [
    {"event": "Luma", "subject": "Arlen", "first": "Mira", "second": "Oren"},
    {"event": "Vela", "subject": "Belen", "first": "Neri", "second": "Tavi"},
    {"event": "Sora", "subject": "Caro", "first": "Pela", "second": "Rumi"},
]
FAMILIES = {
    "tournament": {
        "earlier": (
            "At the fictional {event} Cup, {subject} defeated {first} to reach the semifinal."
        ),
        "target": "In the semifinal of the fictional {event} Cup, {subject} defeated {second}.",
        "question": "Whom did {subject} defeat in the semifinal of the {event} Cup?",
    },
    "interview": {
        "earlier": (
            "To qualify for the final interview at fictional {event}, {subject} "
            "was assessed by {first}."
        ),
        "target": (
            "During the final interview at fictional {event}, {subject} was assessed by {second}."
        ),
        "question": "Who assessed {subject} during the final interview at {event}?",
    },
    "grant": {
        "earlier": (
            "{first}'s proposal reached the final shortlist for the fictional {event} grant."
        ),
        "target": "{second}'s proposal won the fictional {event} grant.",
        "question": "Whose proposal won the {event} grant?",
    },
    "release": {
        "earlier": "Version 2 of the fictional {event} app was announced by {first}.",
        "target": "Version 2 of the fictional {event} app was released by {second}.",
        "question": "Who released version 2 of the {event} app?",
    },
}


def build():
    rows = []
    for family, template in FAMILIES.items():
        for world_index, original in enumerate(WORLDS):
            for swap in (False, True):
                world = dict(original)
                if swap:
                    world["first"], world["second"] = world["second"], world["first"]
                before = template["earlier"].format(**world)
                target = template["target"].format(**world)
                question = template["question"].format(**world)
                conditions = {
                    "target_last": [before, target],
                    "target_first": [target, before],
                    "missing": [before],
                    "target_only": [target],
                }
                for condition, documents in conditions.items():
                    answer = "UNKNOWN" if condition == "missing" else world["second"]
                    text = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(documents))
                    rows.append(
                        {
                            "id": f"{family}:{world_index}:{int(swap)}:{condition}",
                            "family": family,
                            "world": world_index,
                            "swapped": swap,
                            "condition": condition,
                            "answer_groups": [[answer]],
                            "distractor": world["first"],
                            "messages": [
                                {"role": "system", "content": EXPLICIT},
                                {
                                    "role": "user",
                                    "content": f"Documents:\n{text}\n\nQuestion: {question}",
                                },
                            ],
                        }
                    )
    return rows


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rows = build()
    for row in rows:
        row["prompt_tokens"] = len(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
        )
        assert row["prompt_tokens"] + 96 <= 3072
    args.output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(rows, indent=2) + "\n").encode()
    (args.output / "inputs.json").write_bytes(raw)
    protocol = json.loads(
        (Path(__file__).parents[1] / "retrieval_evidence/protocol_v1.json").read_text()
    )
    protocol.pop("pairs")
    protocol.update(
        id="relation-contrast-v1-20261023",
        queries=len(rows),
        maximum_prompt_tokens=max(r["prompt_tokens"] for r in rows),
        inputs_sha256=hashlib.sha256(raw).hexdigest(),
        max_new_tokens=96,
        interpretation={
            "hypothesis": (
                "The observed real semifinal error reflects relation confusion "
                "that also occurs with unfamiliar entities and other event "
                "domains."
            ),
            "contrasts": (
                "Counterbalance candidate identity, target passage order, "
                "target-only evidence and missing-target evidence. Twelve "
                "family/world cells, four families; the 96 prompts are not 96 "
                "independent tasks."
            ),
            "qualification": (
                "At least three family/world cells across at least two families "
                "have a wrong supported answer, while at least 22/24 target-only "
                "outputs have correct answer content. Report missing-target and "
                "order effects separately. This is development headroom only, not "
                "trained benefit."
            ),
            "review": (
                "Read all full outputs, credit semantically correct names with "
                "possessives/explanations; flag truncation. No substring-only "
                "score. Distinguish distractor selection, unsupported answer, "
                "abstention and other error."
            ),
            "limits": (
                "Templates were designed after seeing one tennis failure. No "
                "fresh confirmation or natural distribution claim. If solved, do "
                "not keep increasing noise until a failure appears. No automatic "
                "training follows."
            ),
        },
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {"queries": len(rows), "maximum_prompt_tokens": protocol["maximum_prompt_tokens"]}
        )
    )


if __name__ == "__main__":
    main()
