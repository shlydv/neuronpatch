"""Exposed development inputs, complete answers and explicit whole-answer scoring."""

import re

SYSTEM = (
    "Complete the statement or answer the question. Reply with only the requested "
    "short answer. Do not explain."
)
COUNTRIES = {
    "United States of America": [
        "United States of America",
        "United States",
        "USA",
        "US",
        "U.S.",
        "the United States",
        "the United States of America",
    ],
    "United Kingdom": [
        "United Kingdom",
        "the United Kingdom",
        "UK",
        "U.K.",
        "Britain",
        "Great Britain",
    ],
    "South Korea": ["South Korea", "Republic of Korea", "the Republic of Korea"],
    "Russia": ["Russia", "Russian Federation", "the Russian Federation"],
}
WORDS = dict(
    enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        )
    )
)
WORDS = {str(k): v for k, v in WORDS.items()}


def messages(row):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}]


def normalize(text):
    value = re.sub(r"\s+", " ", text.strip().casefold()).strip("\"'").rstrip(".!?")
    return value[4:] if value.startswith("the ") else value


def whole_hit(output, references):
    """All observed answer text must match and generation must reach EOS."""
    return (
        output["ended"]
        and not output["unexpected_special"]
        and normalize(output["text"]) in {normalize(r) for r in references}
    )


def answers(values):
    values = values if isinstance(values, list) else [values]
    result = []
    for value in values:
        result.extend(COUNTRIES.get(value, [value]))
        if value in WORDS:
            result.append(WORDS[value])
        if value == "New Delhi":
            result.append("Delhi")
    return list(dict.fromkeys(result))


def prepare(parent):
    """Do not consult model outputs; preserve the unused eight-fact bank."""
    utility = [
        {"id": f"utility-{i}", "prompt": p["prompt"], "answers": answers(p["gold"])}
        for i, p in enumerate(parent["utility"])
    ]
    preserve_indices, validation_indices = {0, 16, 24}, {2, 4, 8, 12, 20, 28}
    cases = []
    new_wordings = {
        0: [
            "Which country does Leonardo DiCaprio hold citizenship in?",
            "Name Leonardo DiCaprio's country of citizenship.",
        ],
        1: [
            "Which country is associated with the Academy Award for Best Picture?",
            "Name the country associated with the Academy Award for Best Picture.",
        ],
    }
    adjustments = []
    for source in parent["cases"]:
        c = source["case"]

        def row(probe, key, reference=None):
            prompt = probe["prompt"]
            if probe.get("subject") == "Kamala":
                prompt = prompt.replace("Kamala", "Kamala Harris")
                adjustments.append("Expanded ambiguous Kamala to Kamala Harris before predictions")
            ref = reference if reference is not None else probe["reference_answers"]
            accepted = answers(ref)
            if probe.get("subject") == "Manchester United F.C.":
                accepted.append("England")
                adjustments.append(
                    "Accepted England as well as United Kingdom for the club "
                    "associated-country question before predictions"
                )
            return {"id": f"case-{c}-{key}", "prompt": prompt, "answers": accepted}

        target = source["target"]
        correction = [
            row({"prompt": p}, f"train-{i}", target) for i, p in enumerate(source["train_prompts"])
        ]
        rephrase = next(p for p in source["edit_probes"] if p["kind"] == "rephrase")
        target_val = [row(rephrase, "target-val", target)]
        fresh = [p["prompt"] for p in source["edit_probes"] if p["kind"] == "fresh_wording"]
        assert len(fresh) == 2
        target_test = [
            row({"prompt": p}, f"target-test-{i}", target)
            for i, p in enumerate([*fresh, *new_wordings[c]])
        ]
        preserve = [
            row(p, f"preserve-{i}")
            for i, p in enumerate(source["calibration"])
            if p["subject"] != "Soviet Union"
        ]
        exposed = list(source["exposed_scope"])
        if c == 1:
            facebook = next(p for p in exposed if p["subject"] == "Facebook")
            preserve.append(row(facebook, "preserve-facebook"))
            exposed.remove(facebook)
            adjustments.append(
                "Excluded ambiguous Soviet Union associated-country training label; "
                "used previously exposed Facebook/United States instead and removed "
                "that probe from other banks"
            )
        preserve.extend(utility[i] for i in sorted(preserve_indices))
        cases.append(
            {
                "case": c,
                "target": target,
                "train": correction,
                "preserve": preserve,
                "target_val": target_val,
                "target_test": target_test,
                "scope_val": [row(p, f"scope-val-{i}") for i, p in enumerate(exposed[:3])],
                "scope_test": [
                    row(p, f"scope-test-{i}") for i, p in enumerate(source["evaluation"])
                ],
                "scope_exposed": [row(p, f"scope-exposed-{i}") for i, p in enumerate(exposed[3:])],
                "utility_val": [utility[i] for i in sorted(validation_indices)],
                "utility_test": [
                    p
                    for i, p in enumerate(utility)
                    if i not in preserve_indices | validation_indices
                ],
            }
        )
    for case in cases:
        rows = [p for name, value in case.items() if isinstance(value, list) for p in value]
        assert len({r["id"] for r in rows}) == len(rows)
        assert len({r["prompt"] for r in rows}) == len(rows)
    return {
        "cases": cases,
        "adjustments": adjustments,
        "scope": (
            "Two exposed development facts, intentionally counterfactual targets; "
            "benchmark references are not independently verified current truths. "
            "Not a paper reproduction or fresh confirmation."
        ),
    }
