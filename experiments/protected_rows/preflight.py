"""Exercise the actual runner on a tiny ordinary Qwen3, without downloads."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import peft
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from neuronpatch.patch import fingerprint

from .helpers import Examples, write_json
from .run import capture_spans, choose_anchors, fit, restore_base, select_rows


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "PREFIX " + messages[-1]["content"]

    def encode(self, text, **kwargs):
        if text.startswith("PREFIX "):
            code = sum(map(ord, text)) % 40
            return [1, 2, code + 3, code + 4, 5]
        return [11, 12]

    def decode(self, tokens, **kwargs):
        return " ".join(str(token) for token in tokens)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("preflight.py", "prototype.py", "run.py", "helpers.py")
    }
    torch.set_num_threads(1)
    torch.manual_seed(123)
    device = torch.device(args.device)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            num_hidden_layers=3,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=101,
            max_position_embeddings=64,
            attention_dropout=0,
            _attn_implementation="eager",
            eos_token_id=0,
            pad_token_id=0,
        )
    ).to(device=device, dtype=torch.float16 if args.device == "cuda" else torch.float32)
    model.eval().requires_grad_(False)
    targets = [f"model.layers.{i}.mlp.up_proj" for i in range(3)]
    original_hash = fingerprint(model)
    backups = {name: model.get_submodule(name).weight.detach().cpu().clone() for name in targets}
    examples = Examples(Tokenizer(), device, generation_tokens=3)
    rows = [
        {
            "id": str(i),
            "article": f"article-{i}",
            "context": f"context {i}",
            "question": f"question {i}",
            "answer": "A",
            "answers": ["A"],
            "kind": "answerable",
        }
        for i in range(4)
    ]
    predictions = [
        {"id": row["id"], "em": 1, "generated_token_ids": [13 + index, 0]}
        for index, row in enumerate(rows[:2])
    ]
    protocol = {
        "anchor_count": 2,
        "minimum_anchors": 2,
        "anchor_max_tokens": 24,
        "anchor_token_budget": 30,
        "training_seed": 123,
        "lora_rank": 1,
        "lora_alpha": 2,
        "gradient_clip": 1,
    }
    anchors = choose_anchors(rows, predictions, examples, protocol)
    spans, geometry = capture_spans(model, targets, anchors, examples)
    selected, _ = select_rows(model, targets, spans, rows, examples, count=4)
    assert fingerprint(model) == original_hash
    before = examples.generate(model, rows[-1])["text"]
    results = {}
    with TemporaryDirectory() as directory:
        for family in ("protected", "unprotected", "random", "lora"):
            try:
                results[family] = fit(
                    model,
                    {"family": family, "name": family, "lr": 0.001},
                    selected,
                    spans,
                    targets,
                    anchors,
                    rows[2:],
                    examples,
                    Path(directory) / family,
                    protocol,
                    lambda: None,
                )
                assert torch.isfinite(
                    model(**examples.inputs(rows[-1]), use_cache=False, logits_to_keep=1).logits
                ).all()
            finally:
                restore_base(model, backups)
            assert fingerprint(model) == original_hash
            assert examples.generate(model, rows[-1])["text"] == before
    result = {
        "status": "passed",
        "scope": "tiny Qwen3 runner plumbing, not model-quality evidence",
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "families": results,
        "geometry": geometry,
        "base_weights_and_generation": "exactly restored after every family",
        "source_sha256": source_hashes,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
