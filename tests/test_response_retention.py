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


def test_combined_optimizer_with_kl_matches_independent_direct_loss():
    from experiments.instruction_boundary.preference_training import fit_preferences, score

    model, pairs, cache = fixture()
    model.requires_grad_(False)
    preserved = [pairs[0].chosen, pairs[0].rejected]
    references = cache_responses(model, preserved, base_identity="tiny-base", max_cache_bytes=10000)
    model.head.requires_grad_(True)
    direct_model = copy.deepcopy(model)
    config = dict(
        steps=3,
        lr=0.001,
        seed=43,
        beta=0.1,
        loss_scale=32.0,
        minimum_loss_scale=1.0,
        preservation_weight=0.7,
        gradient_clip=1.0,
    )
    history = fit_preferences(
        model,
        pairs,
        preserved,
        cache,
        base_identity="tiny-base",
        objective="sft",
        protocol=config,
        preservation_objective="kl",
        retained_responses=references,
    )
    optimizer = torch.optim.Adam(direct_model.head.parameters(), lr=config["lr"])
    for row in history:
        i, j = row["example_indices"]
        sample = preserved[j]
        optimizer.zero_grad(set_to_none=True)
        task = -score(direct_model, pairs[i].chosen) / pairs[i].chosen.completion_mask.sum()
        logits = direct_model(sample.input_ids, sample.attention_mask).logits
        logits = logits[:, response_positions(sample), :]
        q = references[j].reference_logits.softmax(-1)
        cross_entropy = -(q * logits.log_softmax(-1)).sum(-1).mean()
        (task + config["preservation_weight"] * cross_entropy).backward()
        torch.nn.utils.clip_grad_norm_(direct_model.head.parameters(), config["gradient_clip"])
        optimizer.step()
        assert row["preserve_nll"] is None and row["preserve_kl"] is not None
        assert row["forward_calls"] == 2
    torch.testing.assert_close(model.head.weight, direct_model.head.weight, atol=1e-7, rtol=1e-5)
    with pytest.raises(ValueError, match="sample order"):
        fit_preferences(
            model,
            pairs,
            preserved,
            cache,
            base_identity="tiny-base",
            objective="sft",
            protocol=config,
            preservation_objective="kl",
            retained_responses=list(reversed(references)),
        )
