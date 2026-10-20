"""Run the actual training/export paths on a tiny FP16 CUDA Qwen before 4B."""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.protected_rows.helpers import Examples, write_json
from neuronpatch.patch import fingerprint

from .run import fit, primary_losses, references, restore


class Tokens:
    eos_token_id = pad_token_id = 2

    def encode(self, text, **kwargs):
        return {"answer": [11, 12], "UNKNOWN": [10], "retained": [13]}[text]


class TinyExamples(Examples):
    def prefix(self, row):
        return row["prefix"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
        .requires_grad_(False)
    )
    examples = TinyExamples(Tokens(), torch.device("cuda"))
    pair = {
        "pair_id": "tiny",
        "answer_token": 11,
        "primary": [
            {"id": "yes", "prefix": [3, 4, 5], "answer": "answer"},
            {"id": "no", "prefix": [3, 6], "answer": "UNKNOWN"},
        ],
        "invariant": [
            {"id": "keep1", "prefix": [3, 4, 7], "answer": "retained"},
            {"id": "keep2", "prefix": [3, 6, 7], "answer": "retained"},
        ],
    }
    name = "model.layers.0.mlp.up_proj"
    backup = {name: model.get_submodule(name).weight.detach().clone()}
    base_hash = fingerprint(model)
    refs = references(model, [pair], examples, 10, lambda: None)
    protocol = {
        "training_seed": 3,
        "lora_rank": 1,
        "lora_alpha": 2,
        "trainable_parameters": 96,
        "epochs": 2,
        "primary_ce_weight": 0.5,
        "pair_weight": 0.25,
        "control_ce_weight": 0.5,
        "control_kl_weight": 1.0,
        "gradient_clip": 1.0,
    }
    outcomes = []
    for family in ("guided", "random", "lora"):
        config = {"family": family, "objective": "ce+pair", "lr": 0.01, "name": family}
        fit(
            model,
            config,
            {name: [1, 9, 17]},
            [pair],
            refs,
            examples,
            10,
            protocol,
            args.output / family,
            lambda: None,
        )
        assert not torch.equal(model.get_submodule(name).weight, backup[name])
        ce, scores = primary_losses(model, pair, examples, 10)
        assert torch.isfinite(ce) and torch.isfinite(scores).all()
        restore(model, backup)
        assert fingerprint(model) == base_hash
        outcomes.append({"family": family, "finite_loss": float(ce), "exact_restore": True})
    result = {
        "gpu": torch.cuda.get_device_name(),
        "base_dtype": "float16",
        "trainable_dtype": "float32",
        "outcomes": outcomes,
    }
    write_json(args.output / "checks.json", result)
    print("CUDA_PREFLIGHT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
