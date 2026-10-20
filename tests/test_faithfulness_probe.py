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
    assert counts["both_reporting_frames_wrong_with_closed_book_correct"] == 0
    assert counts["world_chooses_fabrication_with_closed_book_correct"] == 1
    assert matrix[0]["context_explicit_rotation_same_content"]


def test_exact_option_content_cannot_hide_corrections_or_label_disagreements():
    from experiments.faithfulness_probe.content import exact_option

    options = ["oxygen", "argon", "helium", "neon"]
    assert exact_option("B. argon", options) == "B"
    assert exact_option("argon.", options) == "B"
    assert exact_option("B. oxygen", options) is None
    assert exact_option("B. argon. However, the correct answer is A.", options) is None
    assert exact_option("B. argon. Scientifically incorrect.", options) is None
    assert exact_option("argon", ["argon", "ARGON", "helium", "neon"]) is None


def test_role_control_preserves_instruction_and_document_bytes():
    from experiments.faithfulness_probe.prepare import FRAMES
    from experiments.faithfulness_probe.role_control import system_control_rows

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(" ".join(m["content"] for m in messages))

    body = "Passage: X.\n\nQuestion: Q?\nA. one\nB. two"
    instruction = FRAMES["world_with_context"]
    original = {
        "id": "one",
        "task": "counterfactual",
        "frame": "world_with_context",
        "order": "original",
        "prompt": instruction + "\n\n" + body,
        "answer": "B",
    }
    row = system_control_rows([original], Tokenizer())[0]
    assert row["messages"] == [
        {"role": "system", "content": instruction},
        {"role": "user", "content": body},
    ]
    assert row["answer"] == "B" and row["original_id"] == "one"
    with pytest.raises(ValueError, match="Instruction bytes"):
        system_control_rows([{**original, "prompt": "different instruction"}], Tokenizer())


def test_completion_review_uses_final_response_and_preserves_frozen_rows():
    from experiments.faithfulness_probe.content_report import completed_predictions

    original = [{"id": "a", "text": "B", "generated_token_ids": [1], "truncated": True}]
    extension = [
        {"id": "a", "text": "B; actually A", "generated_token_ids": [1, 2], "truncated": False}
    ]
    completed = completed_predictions(original, [extension])
    assert completed[0]["text"] == "B; actually A"
    assert completed[0]["response_source"] == "extension1"
    assert original[0]["text"] == "B" and original[0]["truncated"]
    with pytest.raises(ValueError, match="every truncated answer"):
        completed_predictions(original, [[]])
    with pytest.raises(ValueError, match="generated prefix"):
        completed_predictions(original, [[{**extension[0], "generated_token_ids": [9, 2]}]])
