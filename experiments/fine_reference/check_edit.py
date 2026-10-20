"""Compare the complete library workflow with unchanged upstream FiNE on CPU.

Uses a tiny random GPT-J and synthetic tokenizer, not a quality benchmark.
The library uses ordinary HF GPT-J; only the upstream CUDA calls are redirected
inside this test. No modified model class is required by the library workflow.
"""

import argparse
import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from run import COMMIT, load_upstream
from transformers import GPTJConfig, GPTJForCausalLM

from neuronpatch.editing import GPTJEditConfig, fit_gptj_edit
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
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--early-stop-probability", type=float)
    parser.add_argument("--add-eos", action="store_true")
    args = parser.parse_args()
    fine, model_class = load_upstream(args.upstream)
    torch.manual_seed(123)
    config = GPTJConfig(
        n_layer=5, n_embd=32, n_head=4, n_inner=64, rotary_dim=8, n_positions=32, vocab_size=101
    )
    dtype = getattr(torch, args.base_dtype)
    base = model_class(config).to(dtype=dtype).eval()
    reference = deepcopy(base)
    candidate = GPTJForCausalLM(config).to(dtype=dtype).eval()
    candidate.load_state_dict(base.state_dict(), strict=True)
    # Every eligible feature at each position guarantees repeated selections.
    hp = fine.FINEHyperParams(
        epochs=4,
        lr=0.001,
        gamma=0.1,
        alpha=1.0,
        beta=10.0,
        neuron_num=128,
        last_layer=3,
        early_stop_prob=args.early_stop_probability,
        random=args.random,
        add_eos=args.add_eos,
        model_name="EleutherAI/gpt-j-6B",
        rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
    )
    request = {"prompt": "Subject is", "subject": "Subject", "target_new": "new answer"}
    source_log = io.StringIO()
    torch.manual_seed(42)  # apply_fine_to_model seeds before execute_fine.
    with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
        with redirect_stdout(source_log):
            expected_delta, neurons, _ = fine.execute_fine(reference, Tokenizer(), request, hp, 0)
    result = fit_gptj_edit(
        candidate,
        Tokenizer(),
        request["prompt"],
        request["target_new"],
        GPTJEditConfig(
            max_steps=hp.epochs,
            lr=hp.lr,
            alpha=hp.alpha,
            beta=hp.beta,
            gamma=hp.gamma,
            neurons_per_token=hp.neuron_num,
            exclude_last_layers=hp.last_layer,
            early_stop_probability=hp.early_stop_prob,
            selection="random" if args.random else "contribution",
            seed=42,
            add_eos=args.add_eos,
        ),
    )
    edit = result.edit
    assert len(result.history) == source_log.getvalue().count("Request 0 Epoch ")
    torch.testing.assert_close(result.selection.reshape(-1, 2), neurons.long(), atol=0, rtol=0)
    torch.testing.assert_close(edit.delta.detach(), expected_delta, atol=0, rtol=0)
    assert len(edit.targets) > len(set(edit.targets))
    for name, tensor in candidate.state_dict().items():
        torch.testing.assert_close(tensor, base.state_dict()[name], atol=0, rtol=0)
    expected_merged = deepcopy(base)
    with torch.no_grad():
        for row, (name, feature) in enumerate(edit.targets):
            expected_merged.get_submodule(name).weight[:, feature].add_(expected_delta[row])
    ids = torch.tensor([result.input_token_ids])
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "edit"
        edit.save(artifact, metadata={"upstream_commit": COMMIT})
        loaded = OccurrenceColumnEdit.load(deepcopy(candidate), artifact)
        torch.testing.assert_close(loaded.delta.detach(), expected_delta, atol=0, rtol=0)
        torch.testing.assert_close(loaded(ids).logits, edit(ids).logits, atol=0, rtol=0)
        merged = loaded.merged_model()
    for name, tensor in merged.state_dict().items():
        torch.testing.assert_close(tensor, expected_merged.state_dict()[name], atol=0, rtol=0)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in candidate.modules())
    print(
        json.dumps(
            {
                "status": "passed",
                "upstream_commit": COMMIT,
                "adam_steps": result.updates,
                "history_rows": len(result.history),
                "stopped_early": result.stopped_early,
                "selection": "random" if args.random else "contribution",
                "add_eos": args.add_eos,
                "selection_occurrences": len(edit.targets),
                "unique_selected_features": len(set(edit.targets)),
                "delta_dtype": str(edit.delta.dtype),
                "base_dtype": "torch." + args.base_dtype,
                "max_trained_delta_difference": float(
                    (edit.delta.detach() - expected_delta).abs().max()
                ),
                "merged_weights": "exact",
                "compact_round_trip": "exact deltas and logits",
                "base_weights": "unchanged",
                "native_model": "ordinary HF GPTJForCausalLM",
                "scope": (
                    "CPU tiny-model workflow parity, including capture/stopping; no quality claim"
                ),
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
