"""CPU fixture for exact upstream teacher-forced metric; no effectiveness claim."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from upstream_metrics import first_answer, load_metric, locality_agreement


class Batch(dict):
    def to(self, device):
        # Test fixture only: upstream asks for CUDA, synthetic tensors stay on CPU.
        assert device == "cuda:0"
        return self


class Tokenizer:
    pad_token_id = 0

    def __init__(self, padding):
        self.padding = padding

    def encode(self, text):
        return [int(t) for t in text.split()]

    def __call__(self, texts, **kwargs):
        rows = [self.encode(t) for t in texts]
        width = max(map(len, rows))
        rows = [
            ([0] * (width - len(r)) + r if self.padding == "left" else r + [0] * (width - len(r)))
            for r in rows
        ]
        ids = torch.tensor(rows)
        return Batch(input_ids=ids, attention_mask=(ids != 0).long())


class Model:
    def __call__(self, input_ids, **kwargs):
        # Predict the next teacher-forced token correctly except token 4 -> 9.
        labels = torch.cat([input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], 1)
        labels = torch.where(labels == 4, 9, labels)
        logits = torch.zeros(*input_ids.shape, 10)
        logits.scatter_(-1, labels.unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits)


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--upstream", type=Path, required=True)
args = parser.parse_args()
metric = load_metric(args.upstream)
for padding in ("left", "right"):
    hp = SimpleNamespace(max_length=40, padding=padding)
    tok = Tokenizer(padding)
    for prompts, targets, accuracy, predictions in [
        ("1 2", "3 4", [0.5], [[3, 9]]),
        (["1 2", "1"], ["3 4", "5"], [0.5, 1.0], [[3, 9], [5]]),
    ]:
        actual = metric(Model(), tok, hp, prompts, targets, 0)
        assert actual == accuracy, actual
        assert metric(Model(), tok, hp, prompts, targets, 0, locality=True) == predictions
assert first_answer([["first", "second"], ["third"]]) == "first"
# Perfect locality can mean preserving an incorrect prediction.
assert locality_agreement([[3, 9]], [[3, 9]]) == [1.0]
assert locality_agreement([[3, 9]], [[3, 4]]) == [0.5]
print(
    json.dumps(
        {
            "status": "passed",
            "checks": [
                "single and variable-length batches",
                "left/right padding",
                "causal target alignment",
                "partial token accuracy",
                "raw locality predictions",
                "first nested alias",
                "agreement is not correctness",
            ],
        }
    )
)
