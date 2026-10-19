"""Compare suffix projection against full Qwen losses and parameter gradients."""

import copy
import runpy
from pathlib import Path

import pytest
import torch

from neuronpatch.losses import preservation_kl

transformers = pytest.importorskip("transformers")
Qwen2Config = transformers.Qwen2Config
Qwen2ForCausalLM = transformers.Qwen2ForCausalLM

helpers = runpy.run_path(str(Path(__file__).parents[1] / "experiments/real_qa/response_loss.py"))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_suffix_matches_full_causal_ce_kl_and_gradients(device):
    torch.manual_seed(19)
    full = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0,
        )
    ).eval()
    full = full.to(device)
    sliced = copy.deepcopy(full)
    teacher = copy.deepcopy(full).requires_grad_(False)
    with torch.no_grad():
        teacher.lm_head.weight.add_(torch.randn_like(teacher.lm_head.weight) * 0.03)
    inputs = {
        "input_ids": torch.tensor([[0, 0, 7, 8, 9, 10], [3, 4, 5, 6, 11, 12]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor(
            [[-100, -100, -100, -100, -100, 10], [-100, -100, -100, -100, 11, 12]]
        ),
    }
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = full(**inputs)
    with torch.no_grad():
        original = teacher(**inputs).logits
    kl = preservation_kl(output.logits[:, :-1], original[:, :-1], inputs["labels"][:, 1:] != -100)
    (output.loss + 0.7 * kl).backward()
    logits, labels = helpers["response_logits"](sliced, inputs)
    with torch.no_grad():
        ref, _ = helpers["response_logits"](teacher, inputs)
    ce_small = helpers["response_ce"](logits, labels)
    kl_small = preservation_kl(logits, ref, labels != -100)
    torch.testing.assert_close(ce_small, output.loss)
    torch.testing.assert_close(kl_small, kl)
    (ce_small + 0.7 * kl_small).backward()
    for (name, expected), (_, actual) in zip(full.named_parameters(), sliced.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-4, msg=name)
    assert logits.shape == (2, 2, 32)
    assert inputs["labels"].shape == (2, 6)  # No caller mutation.


def test_rejects_non_suffix_supervision():
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[-100, 2, -100]])}
    with pytest.raises(ValueError, match="contiguous"):
        helpers["response_logits"](None, inputs)
