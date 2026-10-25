"""Teacher-to-student KL on identical continuations under different prompts.

An experiment primitive, not a new method or an on-policy SecOPD implementation.
The caller owns semantic prompt pairing and teacher correctness. Both samples
must use the same tokenizer vocabulary, and include the same supplied EOS.
"""

import torch

from neuronpatch.losses import preservation_kl

from .response_retention import response_positions, sample_identity, selected_logits


def continuation_ids(sample):
    response_positions(sample)
    length = sample.input_ids.shape[1]
    if not 1 <= sample.prompt_length < length:
        raise ValueError("A nonempty prompt and continuation are required")
    expected = torch.arange(length, device=sample.input_ids.device) >= sample.prompt_length
    if not sample.attention_mask.all() or not torch.equal(sample.completion_mask[0], expected):
        raise ValueError("An uninterrupted attended continuation is required")
    return sample.input_ids[0, sample.prompt_length :]


def cross_context_loss(model, student_sample, teacher, *, base_identity):
    """Align response prediction positions by continuation offset, not prompt index.

    The reference remains bound to its original clean trajectory. We never rewrite
    that identity to make a different prompt pass a same-context cache check.
    This computes full-vocabulary forward KL, not sampled reverse-KL advantages.
    """
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("Eval-mode student required")
    if teacher.base_identity != base_identity or teacher.sample_sha256 != sample_identity(
        teacher.sample
    ):
        raise ValueError("Teacher identity changed")
    if teacher.reference_logits.requires_grad:
        raise ValueError("Detached teacher required")
    teacher_ids = continuation_ids(teacher.sample)
    student_ids = continuation_ids(student_sample)
    if not torch.equal(teacher_ids.cpu(), student_ids.cpu()):
        raise ValueError("Teacher and student continuation tokens must match exactly")
    edited = selected_logits(model, student_sample)
    reference = teacher.reference_logits.to(device=edited.device)
    mask = torch.ones(edited.shape[:2], dtype=torch.bool, device=edited.device)
    return preservation_kl(edited, reference, mask, direction="reference_to_edited")
