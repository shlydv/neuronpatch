"""The knowledge-editing objective in FiNE's published implementation.

This reproduces its single-sequence token scopes and operations. It is distinct
from the paper's written KL direction and from FiNE's safety-editing objectives.
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class KnowledgeEditLoss:
    total: torch.Tensor
    nll: torch.Tensor
    kl: torch.Tensor
    repetition: torch.Tensor
    norm: torch.Tensor


def knowledge_edit_loss(
    edited_logits: torch.Tensor,
    reference_prefix_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    delta: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 0.1,
) -> KnowledgeEditLoss:
    """Compute the reference editor loss for one unpadded causal sequence.

    edited_logits is [prompt + target length, vocabulary], from a forward pass
    containing the prompt and the entire desired target (no appended EOS unless
    EOS is explicitly part of target_tokens). Target predictions use positions
    [-target_length-1:-1]. The frozen reference covers [: -target_length-1],
    excluding the position predicting the first target token. It is detached.

    NLL is the maximum target-token loss. KL(reference || edited) averages over
    prefix positions. Repetition penalizes the highest probability of any target
    token after the complete target. Norm is the joint L2 norm of the global
    [occurrences, output width] delta matrix. Returned KL/repetition/norm already
    include their coefficients. NLL and repetition logits use FP32, as upstream;
    KL and norm retain their supplied precision. This is not a batched SFT loss.
    """
    if edited_logits.ndim != 2 or not edited_logits.is_floating_point():
        raise ValueError("edited_logits must be floating [sequence, vocabulary]")
    sequence, vocabulary = edited_logits.shape
    if min(sequence, vocabulary) <= 0:
        raise ValueError("logit dimensions must be nonempty")
    if target_tokens.ndim != 1 or target_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_tokens must be a one-dimensional integer tensor")
    target_length = target_tokens.numel()
    prefix = sequence - target_length - 1
    if target_length == 0 or prefix < 1:
        raise ValueError("provide a target and at least one preservation-prefix position")
    if (
        reference_prefix_logits.shape != (prefix, vocabulary)
        or not reference_prefix_logits.is_floating_point()
    ):
        raise ValueError("reference prefix must match the preserved prediction positions")
    if delta.ndim != 2 or delta.numel() == 0 or not delta.is_floating_point():
        raise ValueError("delta must be a nonempty floating occurrence matrix")
    if any(
        t.device.type == "meta"
        for t in (edited_logits, reference_prefix_logits, target_tokens, delta)
    ):
        raise ValueError("all tensors must be materialized")
    if (
        reference_prefix_logits.device != edited_logits.device
        or delta.device != edited_logits.device
    ):
        raise ValueError("logits, reference and delta must share a device")
    for coefficient in (alpha, beta, gamma):
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, Real)
            or not math.isfinite(coefficient)
            or coefficient < 0
        ):
            raise ValueError("loss coefficients must be finite and nonnegative")
    labels = target_tokens.to(device=edited_logits.device, dtype=torch.int64)
    if labels.min() < 0 or labels.max() >= vocabulary:
        raise ValueError("target token outside vocabulary")
    if not all(torch.isfinite(t).all() for t in (edited_logits, reference_prefix_logits, delta)):
        raise ValueError("nonfinite loss inputs")
    nll = F.nll_loss(
        F.log_softmax(edited_logits[-target_length - 1 : -1].float(), dim=-1),
        labels,
        reduction="none",
    ).max()
    kl = (
        F.kl_div(
            F.log_softmax(edited_logits[:prefix], dim=-1),
            F.softmax(reference_prefix_logits.detach(), dim=-1),
            reduction="batchmean",
        )
        * alpha
    )
    repetition = (-torch.log(1 - F.softmax(edited_logits[-1].float(), dim=-1)[labels])).max() * beta
    norm = torch.norm(delta, p=2) * gamma
    total = nll + norm + kl + repetition
    if not torch.isfinite(total):
        raise FloatingPointError(
            "reference objective became nonfinite; inspect logits and precision"
        )
    return KnowledgeEditLoss(total=total, nll=nll, kl=kl, repetition=repetition, norm=norm)
