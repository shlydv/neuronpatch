from copy import deepcopy

import pytest

from experiments.relation_learning.verify_math_baseline import marked_value, verify


class Tokenizer:
    def __len__(self):
        return 10

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2]

    def decode(self, tokens, **kwargs):
        return "reasoning 13\n#### 2"


def fixture():
    data = {
        "splits": {
            "development": [
                {"id": str(i), "messages": [], "response": "#### 2", "final_number": "2"}
                for i in range(128)
            ]
        }
    }
    records = [
        {
            "id": str(i),
            "prompt_ids": [1, 2],
            "tokens": [3, 9],
            "text": "reasoning 13\n#### 2",
            "ended": True,
            "parsed_number": "2",
            "classification": "numeric_correct",
        }
        for i in range(128)
    ]
    complete = {
        "status": "complete",
        "training_updates": 0,
        "examples": 128,
        "eos_token_ids": [9],
        "counts": {"numeric_correct": 128, "numeric_wrong": 0, "needs_review": 0, "incomplete": 0},
    }
    return data, {"max_new_tokens": 512}, records, complete


def test_independent_full_output_check():
    assert verify(*fixture(), Tokenizer(), [9])["verified_examples"] == 128


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", "#### 99"),
        ("tokens", [3, 9, 3]),
        ("prompt_ids", [1, 3]),
        ("classification", "numeric_wrong"),
        ("ended", False),
        ("parsed_number", "13"),
        ("id", "127"),
    ],
)
def test_reject_corrupt_artifacts(field, value):
    args = deepcopy(fixture())
    args[2][0][field] = value
    with pytest.raises(ValueError):
        verify(*args, Tokenizer(), [9])


def test_partial_output_is_not_complete():
    args = fixture()
    args[2].pop()
    with pytest.raises(ValueError, match="full frozen"):
        verify(*args, Tokenizer(), [9])


def test_marker_whitespace_is_not_confused_with_reasoning_or_trailing_text():
    assert marked_value("Reasoning\n####\n2\n") == 2
    assert marked_value("Reasoning #### 2") is None
    assert marked_value("Reasoning\n#### 2\nThen 3") is None
    assert marked_value("#### 2\n#### 2") is None
