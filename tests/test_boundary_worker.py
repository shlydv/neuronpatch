import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.instruction_boundary.run import execute
from experiments.instruction_boundary.verify import verify


def test_worker_and_independent_verifier_distinguish_markers_from_qa_correctness(tmp_path):
    class Tokenizer:
        pad_token_id = 0

        def __len__(self):
            return 10

        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1, 2]]) if "return_tensors" in kwargs else [1, 2]

        def decode(self, tokens, **kwargs):
            return "MARKER_ABCD" if tokens[0] == 3 else "7"

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[9])
        index = 0

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled() and kwargs["pad_token_id"] == 0
            outputs = [[3, 9], [4, 9], [3, 3]]
            row = outputs[self.index]
            self.index += 1
            return torch.tensor([[1, 2, *row]])

    rows = [
        {
            "id": str(i),
            "kind": "injected",
            "family": "redirect",
            "messages": [],
            "marker": "MARKER_ABCD",
            "expected": "7",
        }
        for i in range(3)
    ]
    protocol = {"max_new_tokens": 2, "max_context_tokens": 10, "expected_examples": 3}
    predictions = execute(Model(), Tokenizer(), rows, protocol, tmp_path / "result", lambda: None)
    completion = json.loads((tmp_path / "result/completion.json").read_text())
    result = verify(rows, predictions, completion, Tokenizer(), [9], protocol)
    assert result["counts"] == {"ended": 2, "literal_match": 1, "marker_present": 2}
    for field, value in (
        ("text", "corrupt"),
        ("literal_match", True),
        ("prompt_ids", [2]),
        ("ended", False),
    ):
        bad = deepcopy(predictions)
        bad[0][field] = value
        with pytest.raises(ValueError):
            verify(rows, bad, completion, Tokenizer(), [9], protocol)
    with pytest.raises(ValueError, match="Full frozen"):
        verify(rows, predictions[:-1], completion, Tokenizer(), [9], protocol)
