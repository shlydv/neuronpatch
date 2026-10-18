"""Offline Qwen3 architecture smoke test; random weights and synthetic token IDs.

No pretrained knowledge or natural-language behavior is evaluated here.
Install: pip install -e '.[dev,transformers]'
Run: python examples/fit_tiny_qwen.py
"""

from copy import deepcopy
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from neuronpatch import NeuronPatch, contrastive_indices, mean_activations


def run():
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    original = deepcopy(model)
    target = "model.layers.1.mlp.down_proj"
    # Distinct calibration and training batches, with explicit response-token masks.
    positive = torch.randint(1, 16, (4, 8))
    negative = torch.randint(16, 32, (4, 8))
    response_mask = torch.zeros(4, 8, dtype=torch.bool)
    response_mask[:, -3:] = True
    groups = []
    for ids in (positive, negative):
        groups.append(
            mean_activations(
                model,
                [target],
                [
                    (
                        {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                        response_mask,
                    )
                ],
            )[target]
        )
    selected = contrastive_indices(groups[0], groups[1], 8)
    ids = torch.randint(1, 32, (4, 8))
    labels = ids.clone()
    labels[:, :5] = -100  # assistant-only synthetic supervision; HF shifts labels internally
    patch = NeuronPatch(model, {target: selected})
    optimizer = torch.optim.Adam(patch.trainable_parameters(), lr=0.05)
    initial = model(input_ids=ids, labels=labels).loss.item()
    model.train()
    for _ in range(60):
        optimizer.zero_grad()
        model(input_ids=ids, labels=labels).loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = model(input_ids=ids, labels=labels).loss.item()
        fresh_ids = torch.randint(1, 32, (2, 10))
        expected = model(input_ids=fresh_ids).logits
        merged = patch.merged_model()
        torch.testing.assert_close(
            expected, merged(input_ids=fresh_ids).logits, atol=2e-6, rtol=2e-5
        )
        with TemporaryDirectory() as directory:
            patch.save(f"{directory}/patch", metadata={"model": "random-tiny-qwen3"})
            restored = NeuronPatch.load(original, f"{directory}/patch")
            torch.testing.assert_close(
                expected, restored.model(input_ids=fresh_ids).logits, atol=0, rtol=0
            )
    if final >= initial * 0.95:
        raise RuntimeError("Synthetic token loss did not improve sufficiently")
    return {
        "trainable_parameters": patch.num_trainable_parameters,
        "initial_token_loss": initial,
        "final_token_loss": final,
        "selected_indices": selected,
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(run())
    print("Qwen3 synthetic training, fresh-input merge, and reload checks passed.")
