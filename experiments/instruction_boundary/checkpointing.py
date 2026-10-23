"""Decoder recomputation for deterministic eval-mode adaptation.

Non-reentrant checkpointing supports frozen embeddings and autograd.grad used by
the update observer. No train/eval flags change. Install after discovery hooks
have been removed; generation and other no-grad calls use ordinary forwards.
"""

from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


def enable_decoder_recomputation(model):
    layers = tuple(model.model.layers)
    if not layers or model.training or any(layer.training for layer in layers):
        raise ValueError("Nonempty eval-mode decoder required")
    if any(getattr(layer, "_boundary_recomputation", False) for layer in layers):
        raise ValueError("Decoder recomputation already installed")

    def wrap(forward):
        @wraps(forward)
        def recomputed(*args, **kwargs):
            if not torch.is_grad_enabled():
                return forward(*args, **kwargs)
            if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
                raise ValueError("Gradient-bearing decoder calls must not use a cache")
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)

        return recomputed

    for layer in layers:
        layer.forward = wrap(layer.forward)
        layer._boundary_recomputation = True
