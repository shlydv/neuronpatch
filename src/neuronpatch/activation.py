"""Streaming activation statistics for explicit, token-masked calibration data."""

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn

CalibrationBatch = tuple[Mapping[str, torch.Tensor], torch.Tensor]


def mean_activations(
    model: nn.Module,
    targets: Sequence[str],
    batches: Iterable[CalibrationBatch],
) -> dict[str, torch.Tensor]:
    """Average per-example neuron activations entering each target projection.

    Each batch is (model_inputs, selection_mask). The boolean mask has shape
    [batch, sequence] and explicitly selects calibration tokens (for example,
    assistant response tokens). Every example must select at least one token.
    If attention_mask is supplied, selection must exclude padding. Examples
    have equal weight regardless of selected-token count. Results are CPU float64.

    Uses inference/eval mode temporarily and restores every module's training
    flag, even on error. Hooks are always removed. Targets must execute exactly
    once per batch; cached generation and recurrent/shared modules are unsupported.
    No labels are sent implicitly and no gradients or full activation caches are kept.
    """
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("provide unique target module paths")
    modules = {name: model.get_submodule(name) for name in targets}
    if any(type(module) is not nn.Linear for module in modules.values()):
        raise ValueError("targets must be ordinary nn.Linear projections on an unpatched model")
    totals = {
        name: torch.zeros(module.in_features, dtype=torch.float64)
        for name, module in modules.items()
    }
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    current_mask = None
    calls = {}
    count = 0

    def capture(name):
        def hook(module, inputs):
            calls[name] += 1
            x = inputs[0]
            if x.ndim != 3 or tuple(x.shape[:2]) != tuple(current_mask.shape):
                raise ValueError("projection input must match [batch, sequence, features]")
            mask = current_mask.to(x.device)
            # Gather valid tokens before reduction so padded NaNs cannot contaminate sums.
            for row, valid in zip(x, mask):
                values = row[valid].to(dtype=torch.float64)
                if not torch.isfinite(values).all():
                    raise ValueError("selected activations must be finite")
                totals[name].add_(values.mean(dim=0).cpu())

        return hook

    try:
        model.eval()
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(capture(name)))
        with torch.inference_mode():
            for inputs, mask in batches:
                if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] == 0:
                    raise ValueError("selection_mask must be a nonempty 2D boolean tensor")
                if not mask.any(dim=1).all():
                    raise ValueError("each example must select at least one token")
                if "attention_mask" in inputs:
                    attention = inputs["attention_mask"].to(mask.device)
                    if attention.shape != mask.shape or (mask & ~attention.bool()).any():
                        raise ValueError("selection_mask must exclude attention padding")
                current_mask = mask
                calls = dict.fromkeys(targets, 0)
                model(**inputs)
                if any(value != 1 for value in calls.values()):
                    raise ValueError("each target must execute exactly once per batch")
                count += mask.shape[0]
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    if count == 0:
        raise ValueError("calibration data cannot be empty")
    return {name: total / count for name, total in totals.items()}


def contrastive_indices(
    positive: torch.Tensor,
    negative: torch.Tensor,
    count: int,
) -> list[int]:
    """Rank by absolute difference of group mean activations, breaking ties by index.

    This is an association heuristic, not FiNE's attribution formula or proof of
    causality. Use training/calibration data only, with matched prompt/token roles.
    Keep raw scores separately if direction or magnitude is needed for analysis.
    """
    if positive.ndim != 1 or positive.shape != negative.shape:
        raise ValueError("group means must be same-shaped 1D tensors")
    if type(count) is not int or not 0 < count <= positive.numel():
        raise ValueError("count must be between 1 and the feature width")
    positive, negative = positive.detach().cpu().double(), negative.detach().cpu().double()
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("group means must be finite")
    scores = (positive - negative).abs()
    return scores.argsort(descending=True, stable=True)[:count].sort().values.tolist()
