import json
from types import SimpleNamespace

import torch

from experiments.relation_learning.math_baseline import execute


def test_inference_worker_keeps_format_misses_and_truncation_separate(tmp_path):
    texts = {3: "work\n#### 2", 4: "work\n#### 5", 5: "The answer is 2."}

    class Tokenizer:
        pad_token_id = 9

        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1, 2]])

        def decode(self, tokens, **kwargs):
            return texts[tokens[0]]

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[8, 9])
        index = 0

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled() and not kwargs["do_sample"]
            tokens = [[3, 8], [4, 9], [5, 8], [3, 3]][self.index]
            self.index += 1
            return torch.tensor([[1, 2, *tokens]])

    rows = [{"id": str(i), "messages": [], "final_number": "2"} for i in range(4)]
    result = execute(
        Model(),
        Tokenizer(),
        rows,
        {"max_new_tokens": 2, "max_context_tokens": 10},
        tmp_path / "out",
        lambda: None,
    )
    assert [r["classification"] for r in result] == [
        "numeric_correct",
        "numeric_wrong",
        "needs_review",
        "incomplete",
    ]
    assert json.loads((tmp_path / "out/predictions.json").read_text()) == result
    assert json.loads((tmp_path / "out/completion.json").read_text())["training_updates"] == 0
