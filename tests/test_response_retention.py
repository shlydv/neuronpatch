import copy

import pytest
import torch
from test_preference_training import fixture

from experiments.instruction_boundary.response_retention import (
    cache_responses,
    response_positions,
    retention_loss,
)


def test_unchanged_model_has_zero_kl_and_gradient_with_correct_causal_positions():
    model, pairs, _ = fixture()
    model.requires_grad_(False)
    sample = pairs[0].rejected  # Three response targets, with EOS last.
    assert response_positions(sample).tolist() == [1, 2, 3]
    refs = cache_responses(model, [sample], base_identity="base", max_cache_bytes=10000)
    raw = model(sample.input_ids, sample.attention_mask).logits
    torch.testing.assert_close(refs[0].reference_logits, raw[:, [1, 2, 3], :], rtol=0, atol=0)
    assert refs[0].reference_logits.device.type == "cpu"
    assert not refs[0].reference_logits.requires_grad
    model.head.requires_grad_(True)
    loss = retention_loss(model, refs[0], base_identity="base")
    loss.backward()
    assert abs(loss.item()) < 1e-7
    torch.testing.assert_close(
        model.head.weight.grad, torch.zeros_like(model.head.weight), atol=1e-7, rtol=0
    )


def test_changed_model_gradient_matches_direct_teacher_cross_entropy():
    model, pairs, _ = fixture()
    model.requires_grad_(False)
    sample = pairs[0].rejected
    ref = cache_responses(model, [sample], base_identity="base", max_cache_bytes=10000)[0]
    model.head.requires_grad_(True)
    with torch.no_grad():
        model.head.weight.add_(0.2 * torch.randn_like(model.head.weight))
    control = copy.deepcopy(model)
    loss = retention_loss(model, ref, base_identity="base")
    loss.backward()
    logits = control(sample.input_ids, sample.attention_mask).logits[:, 1:4, :]
    q = ref.reference_logits.softmax(-1)
    # Independent cross-entropy gradient; reference entropy is constant.
    direct = -(q * logits.log_softmax(-1)).sum(-1).mean()
    direct.backward()
    assert loss.item() > 0
    torch.testing.assert_close(
        model.head.weight.grad, control.head.weight.grad, atol=1e-7, rtol=1e-5
    )
    assert model.embedding.weight.grad is None


def test_cache_rejects_mutated_trajectory_wrong_base_trainable_teacher_and_budget():
    model, pairs, _ = fixture()
    sample = pairs[0].chosen
    with pytest.raises(ValueError, match="Frozen"):
        cache_responses(model, [sample], base_identity="base", max_cache_bytes=10000)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="byte limit"):
        cache_responses(model, [sample], base_identity="base", max_cache_bytes=1)
    ref = cache_responses(model, [sample], base_identity="base", max_cache_bytes=10000)[0]
    with pytest.raises(ValueError, match="identity changed"):
        retention_loss(model, ref, base_identity="different")
    sample.input_ids[0, 2] = 7
    with pytest.raises(ValueError, match="identity changed"):
        retention_loss(model, ref, base_identity="base")
