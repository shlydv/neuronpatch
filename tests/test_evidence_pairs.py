"""Guard data-pair construction against leaked answers and shared cache IDs."""

import random
import runpy
from pathlib import Path

import pytest

functions = runpy.run_path(str(Path(__file__).parents[1] / "experiments/evidence_pairs/prepare.py"))
candidate, sentence_spans = (functions[name] for name in ("paragraph_candidate", "sentence_spans"))


class Tokenizer:
    def encode(self, text, **kwargs):
        return [sum(map(ord, word)) for word in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(row["content"] for row in messages)


def paragraph():
    context = (
        "The Meridian Archive keeps records of engineering projects in several coastal towns "
        "and nearby inland agricultural settlements. "
        "The archive's first director was Helena Duarte, "
        "who supervised its opening ceremony in 1986. "
        "The courtyard contains a sandstone fountain, "
        "which visitors use as a meeting point each morning. "
        "A public reading room stores regional maps and photographs, "
        "with tables arranged beside the large windows."
    )
    return {
        "context": context,
        "qas": [
            {
                "id": "test-director",
                "question": "Who was the first director of the Meridian Archive?",
                "is_impossible": False,
                "answers": [
                    {"text": "Helena Duarte", "answer_start": context.index("Helena Duarte")}
                ],
            }
        ],
    }


def test_pair_has_same_question_different_cache_ids_and_checked_edits(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    result = candidate(paragraph(), "Archive", Tokenizer(), random.Random(1), 512)
    supported, missing = (result["variants"][kind] for kind in ("answerable", "missing"))
    assert supported["question"] == missing["question"]
    assert supported["id"] != missing["id"] and supported["pair_id"] == missing["pair_id"]
    assert "Helena Duarte" in supported["context"] and "Helena Duarte" not in missing["context"]
    assert result["control_sentence"] not in supported["context"]
    assert result["control_sentence"] in missing["context"]
    assert result["answer_sentence"] not in missing["context"]


def test_repeated_answer_elsewhere_cannot_be_labeled_missing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    value = paragraph()
    value["context"] = value["context"].replace("regional maps", "Helena Duarte's maps")
    assert candidate(value, "Archive", Tokenizer(), random.Random(1), 512) is None


def test_abbreviation_does_not_create_a_half_removed_parenthesis():
    text = (
        "The documents include regional dialects (e.g. Chakavian and Kajkavian). "
        "Another sentence follows."
    )
    spans = sentence_spans(text)
    assert len(spans) == 2
    assert text[slice(*spans[0])].endswith("Kajkavian).")


def reviewed_pair(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    data = candidate(paragraph(), "Archive", Tokenizer(), random.Random(1), 512)
    decision = {
        "index": 0,
        "pair_id": data["pair_id"],
        "accepted": True,
        "reason": "Director identity is absent after the target removal.",
        "invariant": {
            "question": "What are the reading-room tables beside?",
            "answers": ["the large windows"],
            "evidence_unit": 3,
            "review": "Windows fact remains in both contexts.",
        },
    }
    build = runpy.run_path(
        str(Path(__file__).parents[1] / "experiments/evidence_pairs/finalize.py")
    )["build_split"]
    return data, decision, build


def test_reviewed_split_preserves_contexts_and_creates_four_cache_ids(monkeypatch):
    data, decision, build = reviewed_pair(monkeypatch)
    pair = build([data], [decision], 1, Tokenizer())[0]
    rows = pair["primary"] + pair["invariant"]
    assert len({row["id"] for row in rows}) == 4
    assert [row["kind"] for row in pair["primary"]] == ["answerable", "missing"]
    assert all(row["kind"] == "answerable" for row in pair["invariant"])
    assert [row["context"] for row in pair["primary"]] == [
        row["context"] for row in pair["invariant"]
    ]


def test_review_cannot_skip_unreviewed_candidates(monkeypatch):
    data, decision, build = reviewed_pair(monkeypatch)
    decision["index"] = 1
    with pytest.raises(ValueError, match="Unreviewed"):
        build([data, data], [decision], 1, Tokenizer())


def test_control_evidence_cannot_be_a_removed_unit(monkeypatch):
    data, decision, build = reviewed_pair(monkeypatch)
    decision["invariant"].update(evidence_unit=data["answer_unit"], answers=["Helena Duarte"])
    with pytest.raises(ValueError, match="retained evidence"):
        build([data], [decision], 1, Tokenizer())
