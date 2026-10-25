import copy

import pytest
import torch
from test_preference_training import fixture

from experiments.instruction_boundary.cross_context_retention import cross_context_loss
from experiments.instruction_boundary.response_retention import cache_responses
from neuronpatch.supervision import ResponseExample


def test_different_prompt_lengths_align_same_continuation_and_gradient():
    model, pairs, _ = fixture()
    model.requires_grad_(False)
    clean = pairs[0].rejected
    teacher = cache_responses(model, [clean], base_identity="base", max_cache_bytes=10000)[0]
    # Two extra prompt tokens move every supervised prediction position.
    ids = torch.cat([torch.tensor([[4, 5]]), clean.input_ids], dim=1)
    mask = torch.cat([torch.zeros((1, 2), dtype=torch.bool), clean.completion_mask], dim=1)
    attacked = ResponseExample(
        ids, torch.ones_like(ids, dtype=torch.bool), mask, clean.prompt_length + 2
    )
    model.head.requires_grad_(True)
    with torch.no_grad():
        model.head.weight.add_(torch.randn_like(model.head.weight) * 0.1)
    direct_model = copy.deepcopy(model)
    loss = cross_context_loss(model, attacked, teacher, base_identity="base")
    loss.backward()
    values = direct_model(ids, attacked.attention_mask).logits[:, attacked.prompt_length - 1 : -1]
    q = teacher.reference_logits.softmax(-1)
    expected = (q * (q.log() - values.log_softmax(-1))).sum(-1).mean()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(model.head.weight.grad, direct_model.head.weight.grad)
    assert teacher.reference_logits.grad is None
    # Same length but different target tokens must not silently align.
    ids[0, -1] = (ids[0, -1] + 1) % values.shape[-1]
    with pytest.raises(ValueError, match="tokens must match"):
        cross_context_loss(model, attacked, teacher, base_identity="base")


def test_reject_changed_teacher_and_gaps_in_continuation():
    model, pairs, _ = fixture()
    model.requires_grad_(False)
    clean = pairs[0].rejected
    teacher = cache_responses(model, [clean], base_identity="base", max_cache_bytes=10000)[0]
    student = copy.deepcopy(clean)
    student.completion_mask[0, -2] = False
    with pytest.raises(ValueError, match="uninterrupted"):
        cross_context_loss(model, student, teacher, base_identity="base")
    clean.input_ids[0, 0] += 1
    with pytest.raises(ValueError, match="identity"):
        cross_context_loss(model, pairs[0].chosen, teacher, base_identity="base")
