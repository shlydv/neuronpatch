"""Compare library occurrence updates with unchanged upstream FiNE training on CPU.

Uses a tiny random GPT-J and synthetic tokenizer; this is numerical integration
verification, not a behavioral replication or a GPU/mixed-FP16 quality result.
Only upstream Tensor.cuda calls are redirected to CPU inside this test process.
"""

import argparse
import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from run import COMMIT, load_upstream
from transformers import GPTJConfig

from neuronpatch.contribution import contribution_indices
from neuronpatch.knowledge import knowledge_edit_loss
from neuronpatch.occurrence import OccurrenceColumnEdit


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        tokens = [7, 8] if text.strip() == "new answer" else [3, 4, 5]
        return torch.tensor([tokens]) if return_tensors else tokens

    def batch_decode(self, ids, **kwargs):
        return [str(row.tolist()) for row in ids]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--base-dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    fine, model_class = load_upstream(args.upstream)
    torch.manual_seed(123)
    config = GPTJConfig(
        n_layer=5, n_embd=32, n_head=4, n_inner=64, rotary_dim=8, n_positions=32, vocab_size=101
    )
    base = model_class(config).to(dtype=getattr(torch, args.base_dtype)).eval()
    reference, candidate = deepcopy(base), deepcopy(base)
    # Select all eligible features at both target positions to guarantee repeats.
    hp = fine.FINEHyperParams(
        epochs=4,
        lr=0.001,
        gamma=0.1,
        alpha=1.0,
        beta=10.0,
        neuron_num=128,
        last_layer=3,
        early_stop_prob=None,
        model_name="EleutherAI/gpt-j-6B",
        rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
    )
    request = {"prompt": "Subject is", "subject": "Subject", "target_new": "new answer"}
    with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
        with redirect_stdout(io.StringIO()):
            expected_delta, neurons, _ = fine.execute_fine(reference, Tokenizer(), request, hp, 0)
    ids = torch.tensor([[0, 3, 4, 5, 7, 8]])
    labels = torch.tensor([7, 8])
    candidate.set_get_activations()
    with torch.no_grad():
        baseline = candidate(ids, attention_mask=torch.ones_like(ids)).logits
    candidate.cal_activations = False
    activations = torch.stack(candidate.activations).permute(1, 2, 0, 3)[0, -3:-1]
    supports = contribution_indices(
        activations,
        [layer.mlp.fc_out.weight for layer in candidate.transformer.h],
        candidate.lm_head.weight,
        baseline[0, -3:-1].argmax(-1),
        hp.neuron_num,
        exclude_last_layers=hp.last_layer,
    ).reshape(-1, 2)
    torch.testing.assert_close(supports, neurons, atol=0, rtol=0)
    targets = [
        (hp.rewrite_module_tmp.format(layer), feature) for layer, feature in supports.tolist()
    ]
    assert len(targets) > len(set(targets))
    edit = OccurrenceColumnEdit(candidate, targets, delta_dtype=torch.bfloat16)
    original = baseline[0, :-3].detach()
    optimizer = torch.optim.Adam(edit.trainable_parameters(), lr=hp.lr)
    for _ in range(hp.epochs):
        logits = edit(ids, attention_mask=torch.ones_like(ids)).logits[0]
        loss = knowledge_edit_loss(
            logits, original, labels, edit.delta, alpha=hp.alpha, beta=hp.beta, gamma=hp.gamma
        )
        loss.total.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.testing.assert_close(edit.delta.detach(), expected_delta, atol=0, rtol=0)
    # The editor freezes but does not mutate its base weights during training.
    for name, tensor in candidate.state_dict().items():
        torch.testing.assert_close(tensor, base.state_dict()[name], atol=0, rtol=0)
    expected_merged = deepcopy(base)
    with torch.no_grad():
        for row, (name, feature) in enumerate(targets):
            expected_merged.get_submodule(name).weight[:, feature].add_(expected_delta[row])
    merged = edit.merged_model()
    for name, tensor in merged.state_dict().items():
        torch.testing.assert_close(tensor, expected_merged.state_dict()[name], atol=0, rtol=0)
    assert not any(module._forward_hooks for module in candidate.modules())
    print(
        json.dumps(
            {
                "status": "passed",
                "upstream_commit": COMMIT,
                "adam_steps": hp.epochs,
                "selection_occurrences": len(targets),
                "unique_selected_features": len(set(targets)),
                "delta_dtype": str(edit.delta.dtype),
                "base_dtype": "torch." + args.base_dtype,
                "max_trained_delta_difference": float(
                    (edit.delta.detach() - expected_delta).abs().max()
                ),
                "merged_weights": "exact",
                "base_weights": "unchanged",
                "scope": (
                    "CPU tiny-model selector/objective/training/merge parity; "
                    "no model-quality claim"
                ),
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
