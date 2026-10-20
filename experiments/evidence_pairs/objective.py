"""Experimental first-token evidence separation, outside the public API.

Pair order is [supported, missing]. A pair uses the SAME question and the SAME
answer token under both contexts. This objective tests conditional adaptation;
it is ordinary pairwise ranking with a reference-centered regularizer, not a
new ranking algorithm or a guarantee of correct generated answers.
"""

import math

import torch
from torch.nn import functional as F


def evidence_scores(logits, answer_tokens, *, unknown_token):
    """Return z_UNKNOWN - z_answer for [pairs, 2, vocabulary] first-token logits.

    Unlike UNKNOWN-versus-all log odds, this fixed within-pair token comparison
    cancels ANY context-independent vocabulary bias in the pair's score difference.
    The first answer token is only a proxy for the full answer; full continuations
    must be evaluated separately. Floating inputs are promoted to at least FP32.
    """
    if (
        logits.ndim != 3
        or logits.shape[0] == 0
        or logits.shape[1] != 2
        or logits.shape[2] < 2
        or not logits.is_floating_point()
    ):
        raise ValueError("finite floating first-token logits [pairs, 2, V], V>=2 required")
    if (
        type(unknown_token) is not int
        or not 0 <= unknown_token < logits.shape[2]
        or answer_tokens.dtype != torch.long
        or answer_tokens.shape != (logits.shape[0],)
        or answer_tokens.device != logits.device
        or (answer_tokens < 0).any()
        or (answer_tokens >= logits.shape[2]).any()
        or (answer_tokens == unknown_token).any()
    ):
        raise ValueError("one distinct, valid answer token per pair on the logits device required")
    if not torch.isfinite(logits).all():
        raise ValueError("first-token logits must be finite")
    values = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
    answer = values.gather(-1, answer_tokens[:, None, None].expand(-1, 2, 1)).squeeze(-1)
    return values[..., unknown_token] - answer


def separation_loss(scores, reference_scores, *, margin=4.0, temperature=1.0, center_weight=0.1):
    """Rank missing above supported while penalizing drift of each pair's center.

    L = mean softplus((margin - (s_missing - s_supported)) / temperature)
        + center_weight * mean((mean_pair(s) - mean_pair(s_reference))**2).

    The ranking term is invariant to adding an arbitrary shared vocabulary bias to
    each pair's logits. From the reference model, such a bias cannot lower this total
    loss: it leaves ranking unchanged and adds a nonnegative center penalty. The
    reference is always detached. This is not true of arbitrary conditional weight
    updates, nor a proof of semantic preservation. Margin/temperature are in logits.
    """
    if (
        scores.ndim != 2
        or scores.shape[0] == 0
        or scores.shape[1] != 2
        or reference_scores.shape != scores.shape
        or not scores.is_floating_point()
        or not reference_scores.is_floating_point()
        or reference_scores.device != scores.device
        or not torch.isfinite(scores).all()
        or not torch.isfinite(reference_scores).all()
    ):
        raise ValueError("finite floating [pairs, 2] scores and matching references required")
    if (
        any(type(value) not in (float, int) for value in (margin, temperature, center_weight))
        or not all(math.isfinite(value) for value in (margin, temperature, center_weight))
        or margin < 0
        or temperature <= 0
        or center_weight < 0
    ):
        raise ValueError("finite margin>=0, temperature>0, center_weight>=0 required")
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    values = scores.to(dtype)
    reference = reference_scores.detach().to(dtype)
    gap = values[:, 1] - values[:, 0]
    ranking = F.softplus((margin - gap) / temperature).mean()
    centers = (values.mean(-1) - reference.mean(-1)).square().mean()
    return ranking + center_weight * centers
