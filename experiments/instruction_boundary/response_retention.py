"""Exact response-position reference caches for a future retention control.

This is reference-distribution retention, not reference-answer reinforcement.
The caller chooses the trajectories (and owns their quality/provenance), measures
base identity, and decides the loss weight. No output-quality guarantee is made.
"""

from dataclasses import dataclass

import torch

from neuronpatch.losses import preservation_kl
from neuronpatch.supervision import ResponseExample

from .preference_training import digest


def response_positions(sample):
    ids, attention, completion = sample.input_ids, sample.attention_mask, sample.completion_mask
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or ids.shape[1] < 2
        or ids.dtype != torch.long
        or any(
            m.shape != ids.shape or m.dtype != torch.bool or m.device != ids.device
            for m in (attention, completion)
        )
    ):
        raise ValueError("One response with aligned bool masks and int64 IDs required")
    if (
        completion[:, 0].any()
        or (completion & ~attention).any()
        or (completion[:, 1:] & ~attention[:, :-1]).any()
        or not completion.any()
    ):
        raise ValueError("Response targets require attended preceding prediction positions")
    # Target token t is predicted by logits at t-1, including the supplied EOS.
    return completion[0, 1:].nonzero(as_tuple=True)[0]


def sample_identity(sample):
    return digest(
        {
            "input_ids": sample.input_ids.tolist(),
            "attention_mask": sample.attention_mask.tolist(),
            "completion_mask": sample.completion_mask.tolist(),
            "prompt_length": sample.prompt_length,
        }
    )


@dataclass(frozen=True)
class RetainedResponse:
    sample: ResponseExample
    sample_sha256: str
    base_identity: str
    reference_logits: torch.Tensor


def selected_logits(model, sample):
    positions = response_positions(sample)
    logits = model(
        input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
    ).logits
    return logits[:, positions, :]


def cache_responses(model, samples, *, base_identity, max_cache_bytes, check_time=lambda: None):
    """Cache only response-prediction logits on CPU, preserving their original dtype.

    The caller must pass a measured base fingerprint. This check cannot establish
    that a caller-supplied identity is true. Cache construction precedes adapters.
    Reject rather than truncate or quantize when the explicit host-memory cap is hit.
    """
    if (
        model.training
        or any(m.training for m in model.modules())
        or any(p.requires_grad for p in model.parameters())
    ):
        raise ValueError("Frozen eval-mode reference model required")
    if not samples or not isinstance(base_identity, str) or not base_identity:
        raise ValueError("Nonempty response bank and measured base identity required")
    if type(max_cache_bytes) is not int or max_cache_bytes < 1:
        raise ValueError("Positive cache byte limit required")
    cache, size = [], 0
    with torch.no_grad():
        for sample in samples:
            check_time()
            logits = selected_logits(model, sample).detach().cpu().contiguous()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite reference logits")
            size += logits.numel() * logits.element_size()
            if size > max_cache_bytes:
                raise ValueError("Response reference cache exceeds byte limit")
            cache.append(RetainedResponse(sample, sample_identity(sample), base_identity, logits))
    return cache


def retention_loss(model, retained, *, base_identity):
    """Mean full-vocabulary KL(reference || edited), response positions only."""
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("Eval-mode edited model required")
    if retained.base_identity != base_identity or retained.sample_sha256 != sample_identity(
        retained.sample
    ):
        raise ValueError("Reference base or tokenized trajectory identity changed")
    if retained.reference_logits.requires_grad:
        raise ValueError("Reference logits must remain detached")
    edited = selected_logits(model, retained.sample)
    reference = retained.reference_logits.to(device=edited.device)
    mask = torch.ones(edited.shape[:2], dtype=torch.bool, device=edited.device)
    return preservation_kl(edited, reference, mask, direction="reference_to_edited")
