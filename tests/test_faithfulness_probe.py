"""Guard answer parsing, no label leakage and rotated-option controls."""

import pytest

from experiments.faithfulness_probe.prepare import mcq_prompt, remap
from experiments.faithfulness_probe.run import decision, messages


@pytest.mark.parametrize(
    "text,want",
    [("A", "A"), ("(B)", "B"), ("C.", "C"), ("Answer: D", None), ("A or B", None), ("AB", None)],
)
def test_letter_decisions_require_one_complete_choice(text, want):
    assert decision(text, kind="mcq") == want


@pytest.mark.parametrize(
    "text,want",
    [
        ("unknown.", "UNKNOWN"),
        ("CONFLICT", "CONFLICT"),
        ("not unknown", None),
        ("no conflict", None),
    ],
)
def test_no_substring_abstention_false_positives(text, want):
    assert decision(text, kind="decision") == want


def test_option_permutation_moves_labels_with_text_without_gold_leakage():
    row = {
        "question": "Which gas?",
        "context": "The passage claims argon.",
        "choices": {"label": list("ABCD"), "text": ["oxygen", "argon", "helium", "neon"]},
        "answerKey": "SECRET_GOLD",
        "justification": "SECRET_REASON",
    }
    text = mcq_prompt(row, "context_standard", [1, 2, 3, 0])
    assert "A. argon" in text and "D. oxygen" in text
    assert "SECRET_GOLD" not in text and "SECRET_REASON" not in text
    assert remap("B", list("ABCD"), [1, 2, 3, 0]) == "A"
    assert remap("A", list("ABCD"), [1, 2, 3, 0]) == "D"


def test_closed_book_has_no_document_and_message_path_has_no_targets():
    row = {"question": "Q?", "context": "SECRET_DOCUMENT", "choices": {"text": list("wxyz")}}
    text = mcq_prompt(row, "closed_book", [0, 1, 2, 3])
    assert "SECRET_DOCUMENT" not in text
    assert messages({"prompt": text, "answer": "SECRET_TARGET"}) == [
        {"role": "user", "content": text}
    ]
    explicit = [{"role": "user", "content": "actual input"}]
    assert messages({"messages": explicit, "answer": "SECRET_TARGET"}) == explicit


def test_analysis_separates_prompt_repair_from_science_interference():
    from experiments.faithfulness_probe.analyze import summarize

    rows = []
    # Truth is A, supplied claim is B. Standard repeats truth; the explicit
    # reporting instruction works, but the science-with-context mode is fooled.
    for frame, parsed, correct in [
        ("closed_book", "A", True),
        ("context_standard", "A", False),
        ("context_explicit", "B", True),
        ("world_with_context", "B", False),
    ]:
        rows.append(
            {
                "source_id": "one",
                "task": "counterfactual",
                "frame": frame,
                "order": "original",
                "claim_letter": "B",
                "truth_letter": "A",
                "parsed": parsed,
                "text": parsed,
                "correct": correct,
                "truncated": False,
            }
        )
    # Rotated A denotes original B, so it agrees in content with explicit B.
    rows.append(
        {
            **rows[2],
            "order": "rotated",
            "parsed": "A",
            "text": "A",
            "claim_letter": "A",
            "truth_letter": "D",
        }
    )
    summary, matrix = summarize(rows)
    counts = summary["counterfactual_counts"]
    assert counts["questions"] == 1 and summary["primary_query_count"] == 5
    assert counts["standard_wrong_explicit_fixed_with_closed_book_correct"] == 1
    assert counts["world_chooses_fabrication_with_closed_book_correct"] == 1
    assert matrix[0]["context_explicit_rotation_same_content"]
