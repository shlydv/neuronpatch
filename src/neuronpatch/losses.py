"""Explicit causal response supervision and masked distribution preservation."""

import torch
from torch.nn import functional as F


def completion_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Causal teacher-forced NLL over every explicitly selected response token.

    All masks/IDs are [B,T]; logits are [B,T,V]. completion_mask marks TARGET
    token positions, unlike preservation_kl's prediction-position mask. This
    function shifts exactly once: logits[:,t-1] predict input_ids[:,t]. Include
    the assistant's end-of-turn/EOS token in the completion mask to supervise
    stopping. Chat templates and tokenization remain caller-owned; there is no
    implicit EOS, truncation or token-boundary guess.

    Tokens are averaged within each response, then examples equally. Selected
    targets must be attended and have an attended preceding context token. No
    direct loss is applied to prompt or padding logits. FP16/BF16 logits are
    promoted to FP32. This is ordinary SFT, not a new loss or a guarantee about
    free-running answers.
    """
    if (
        logits.ndim != 3
        or not logits.is_floating_point()
        or logits.shape[0] == 0
        or logits.shape[1] < 2
        or logits.shape[2] < 2
        or input_ids.dtype != torch.long
        or input_ids.shape != logits.shape[:2]
        or input_ids.device != logits.device
    ):
        raise ValueError("floating [B,T,V] logits and same-device int64 [B,T] IDs required")
    if any(
        mask.dtype != torch.bool or mask.shape != input_ids.shape or mask.device != logits.device
        for mask in (completion_mask, attention_mask)
    ):
        raise ValueError("same-device bool [B,T] completion and attention masks required")
    if (
        completion_mask[:, 0].any()
        or (completion_mask & ~attention_mask).any()
        or (completion_mask[:, 1:] & ~attention_mask[:, :-1]).any()
        or not completion_mask[:, 1:].any(dim=1).all()
    ):
        raise ValueError("every response target needs an attended preceding token; exclude padding")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    losses = []
    for values, ids, selected in zip(logits[:, :-1], input_ids[:, 1:], completion_mask[:, 1:]):
        values, ids = values[selected].to(dtype), ids[selected]
        if not torch.isfinite(values).all() or (ids < 0).any() or (ids >= logits.shape[-1]).any():
            raise ValueError(
                "selected prediction logits must be finite and target IDs in vocabulary"
            )
        losses.append(F.cross_entropy(values, ids))
    return torch.stack(losses).mean()


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
