"""Standard DPO primitives for a controlled boundary-objective experiment.

Experimental, not a library default or a novel objective. References must come
from the frozen original model on the same complete responses. The caller owns
reference provenance, pair weights, adapter lifecycle, and preservation losses.
"""

import math

import torch
from torch.nn import functional as F

from neuronpatch.losses import completion_nll
from neuronpatch.supervision import ResponseExample


def response_log_probability(logits: torch.Tensor, sample: ResponseExample) -> torch.Tensor:
    """Sum causal log probabilities of one full response, including supplied EOS.

    Reuse the audited causal shift and mask validation. Unlike SFT's mean token
    NLL, ordinary DPO uses sequence sums. Restrict to one response to avoid
    accidentally multiplying an average over unequal-length batch examples.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Exactly one response is required")
    mean_nll = completion_nll(
        logits, sample.input_ids, sample.completion_mask, sample.attention_mask
    )
    return -mean_nll * sample.completion_mask.sum()


def dpo_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Reference-relative logistic preference loss; no length normalization.

    Positive gradients on the rejected log probability lower that probability
    under gradient descent. Reference tensors must already be detached, rather
    than silently hiding a reference-model lifecycle mistake.
    """
    scores = (chosen, rejected, reference_chosen, reference_rejected)
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError("Finite positive beta required")
    if any(
        not isinstance(s, torch.Tensor)
        or s.ndim != 0
        or not s.is_floating_point()
        or not torch.isfinite(s)
        for s in scores
    ):
        raise ValueError("Finite scalar log probabilities required")
    if any(s.device != chosen.device for s in scores):
        raise ValueError("Scores must share a device")
    if reference_chosen.requires_grad or reference_rejected.requires_grad:
        raise ValueError("Frozen reference scores must not require gradients")
    dtype = torch.float64 if any(s.dtype == torch.float64 for s in scores) else torch.float32
    c, r, rc, rr = (s.to(dtype) for s in scores)
    margin = beta * ((c - r) - (rc - rr))
    return -F.logsigmoid(margin)
