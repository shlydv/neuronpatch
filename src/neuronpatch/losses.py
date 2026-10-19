"""Explicitly masked distribution preservation; no implicit causal-token shifting."""

import torch
from torch.nn import functional as F


def preservation_kl(
    edited_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    direction: str = "reference_to_edited",
    token_reduction: str = "mean",
) -> torch.Tensor:
    """Per-example KL on selected prediction positions, in nats.

    Logits must both be [batch, sequence, vocabulary] on the same device. A bool
    [batch, sequence] mask selects LOGIT positions, not target-token positions.
    For a causal LM, logit t predicts token t+1: callers own alignment and must
    exclude padding. Each example needs at least one selected position. Tokens
    are averaged within each example by default. With token_reduction="max",
    use each example's largest token KL instead. Examples are then averaged
    equally (not length-weighted). Max reduction splits gradients evenly at ties.
    Max is an experimental objective, not a guarantee of semantic preservation.

    Reference logits are always detached. Default is KL(reference || edited),
    matching PyTorch kl_div(log_edited, reference). The reverse option explicitly
    computes KL(edited || reference). Neither direction guarantees preservation
    outside the supplied contexts. Inputs are promoted to float32 unless double.
    Only selected positions must be finite; masked padding may contain NaNs.
    """
    if direction not in ("reference_to_edited", "edited_to_reference"):
        raise ValueError("unknown KL direction")
    if token_reduction not in ("mean", "max"):
        raise ValueError("unknown token reduction")
    if (
        edited_logits.ndim != 3
        or edited_logits.shape != reference_logits.shape
        or edited_logits.shape[0] == 0
        or edited_logits.shape[-1] < 2
        or not edited_logits.is_floating_point()
        or not reference_logits.is_floating_point()
    ):
        raise ValueError("logits must be same-shaped nonempty floating [B,T,V], V>=2")
    if (
        mask.dtype != torch.bool
        or mask.shape != edited_logits.shape[:2]
        or mask.device != edited_logits.device
        or reference_logits.device != edited_logits.device
        or not mask.any(dim=1).all()
    ):
        raise ValueError("mask must match logits/device and select tokens in every example")
    dtype = torch.float64 if edited_logits.dtype == torch.float64 else torch.float32
    losses = []
    for edited, reference, valid in zip(edited_logits, reference_logits.detach(), mask):
        edited = edited[valid].to(dtype)
        reference = reference[valid].to(dtype)
        if not torch.isfinite(edited).all() or not torch.isfinite(reference).all():
            raise ValueError("selected logits must be finite")
        log_edited = F.log_softmax(edited, dim=-1)
        log_reference = F.log_softmax(reference, dim=-1)
        if direction == "reference_to_edited":
            terms = log_reference.exp() * (log_reference - log_edited)
        else:
            terms = log_edited.exp() * (log_edited - log_reference)
        token_losses = terms.sum(dim=-1)
        losses.append(token_losses.mean() if token_reduction == "mean" else token_losses.amax())
    return torch.stack(losses).mean()
