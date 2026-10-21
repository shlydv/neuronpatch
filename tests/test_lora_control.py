"""Exercise the official PEFT baseline on actual tiny GPT-J modules."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from experiments.fine_transfer.lora_control import (
    attach_gptj_lora,
    load_gptj_lora,
    merge_gptj_lora,
    remove_gptj_lora,
)


def tiny_model():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    torch.manual_seed(413)
    return transformers.GPTJForCausalLM(
        transformers.GPTJConfig(
            vocab_size=19,
            n_positions=32,
            n_embd=16,
            n_layer=4,
            n_head=2,
            n_inner=24,
            rotary_dim=8,
        )
    ).eval()


@pytest.mark.parametrize("scope,count,parameters", [("down", 4, 320), ("attention_mlp", 24, 1664)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_broad_target_coverage_budget_and_zero_initialization(scope, count, parameters, dtype):
    model = tiny_model().to(dtype)
    ids = torch.tensor([[0, 3, 8, 4]])
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits.clone()
    state = torch.random.get_rng_state().clone()
    adapted, report = attach_gptj_lora(model, scope=scope)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(report["targets"]) == count and report["trainable_scalars"] == parameters
    assert any("transformer.h.3." in name for name in report["targets"])
    with torch.no_grad():
        actual = adapted(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_official_adapter_training_export_reload_and_merge(tmp_path, dtype):
    model = tiny_model().to(dtype)
    fresh = deepcopy(model)
    original = [(p, p.detach().clone()) for p in model.parameters()]
    adapted, report = attach_gptj_lora(model)
    trainable = [p for p in adapted.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=0.01)
    ids = torch.tensor([[0, 3, 8, 4]])
    before = adapted(input_ids=ids, use_cache=False).logits.detach().clone()
    for _ in range(3):
        logits = adapted(input_ids=ids, use_cache=False).logits
        F.cross_entropy(logits[0, -1].unsqueeze(0), torch.tensor([7])).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for p, value in original:
        assert (
            p.dtype == value.dtype
            and torch.equal(p, value)
            and not p.requires_grad
            and p.grad is None
        )
    with torch.no_grad():
        edited = adapted(input_ids=ids, use_cache=False).logits.clone()
    assert not torch.equal(edited, before)
    adapted.save_pretrained(
        tmp_path / "adapter", safe_serialization=True, save_embedding_layers=False
    )
    reloaded = load_gptj_lora(fresh, tmp_path / "adapter")
    factors = {n: p for n, p in adapted.named_parameters() if "lora_" in n}
    for n, p in reloaded.named_parameters():
        if n in factors:
            assert p.dtype == torch.float32 and torch.equal(p, factors[n])
    with torch.no_grad():
        restored = reloaded(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(edited, restored, atol=0, rtol=0)
    merged = merge_gptj_lora(reloaded)
    assert all(p.dtype == dtype for p in merged.parameters())
    with torch.no_grad():
        merged_logits = merged(input_ids=ids, use_cache=False).logits
    tolerance = 1e-6 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(edited, merged_logits, atol=tolerance, rtol=tolerance)
    assert not any("lora_" in name for name, _ in merged.named_parameters())
    assert report["scope"] == "attention_mlp"


def test_repeated_attachment_and_exact_restoration_after_merging():
    from experiments.adapter_comparison.run import restore_weights, snapshot_weights
    from experiments.fine_transfer.lora_control import gptj_targets
    from neuronpatch.patch import fingerprint

    model = tiny_model().half().requires_grad_(False)
    identity = fingerprint(model)
    snapshot = snapshot_weights(model, gptj_targets(model, "attention_mlp"))
    for merge in (False, True, False):
        adapted, _ = attach_gptj_lora(model)
        with torch.no_grad():
            for name, p in adapted.named_parameters():
                if "lora_B" in name:
                    p.fill_(0.0137)
        if merge:
            model = merge_gptj_lora(adapted)
            assert fingerprint(model) != identity
            restore_weights(model, snapshot)
        else:
            model = remove_gptj_lora(adapted)
        assert fingerprint(model) == identity
        assert not hasattr(model, "peft_config")
        assert all(not p.requires_grad for p in model.parameters())
