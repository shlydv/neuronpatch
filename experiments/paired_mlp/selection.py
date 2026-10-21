"""Full-response gradient factors for a simple paired-neuron ranking control.

This is group gradient magnitude, not causal discovery or a new attribution
algorithm. It ranks a local first-order direction, not Adam's actual updates or
future answer quality. No evaluation outputs or fitted adapter are consumed.
"""

import math

import torch
from torch import nn

from experiments.fine_transfer.gradient_controls import gradient_statistics
from neuronpatch.losses import completion_nll


def capture_response_factors(model, example, names, *, loss_scale=128.0):
    """Capture [all tokens,input/output] factors of one complete-response loss.

    The base must be frozen and in eval mode. Embedding outputs become a local
    differentiable leaf; base parameter gradients are never enabled. Each target
    must execute once, so this helper is not for reused/fused/distributed layers.
    Constant scaling mitigates FP16 derivative underflow and is undone in FP32.
    Returned factors are detached CPU tensors. Hooks are removed on every exit.
    """
    names = tuple(names)
    if (
        model.training
        or any(p.requires_grad for p in model.parameters())
        or not torch.is_grad_enabled()
        or not names
        or len(set(names)) != len(names)
        or example.input_ids.ndim != 2
        or example.input_ids.shape[0] != 1
    ):
        raise ValueError(
            "single-example supervision, unique targets and eval-mode frozen base required"
        )
    if isinstance(loss_scale, bool) or not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("finite positive loss_scale required")
    modules = {name: model.get_submodule(name) for name in names}
    if any(type(module) is not nn.Linear for module in modules.values()):
        raise ValueError("ordinary Linear projections required")
    activations, outputs, handles = {}, {}, []

    def capture(name):
        def hook(module, args, output):
            if name in outputs or args[0].shape[:2] != example.input_ids.shape:
                raise ValueError("each projection must execute once over the complete sequence")
            activations[name] = args[0][0].detach().float().cpu()
            output.retain_grad()
            outputs[name] = output

        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))
        embedded = model.get_input_embeddings()(example.input_ids).detach().requires_grad_(True)
        logits = model(
            inputs_embeds=embedded, attention_mask=example.attention_mask, use_cache=False
        ).logits
        loss = completion_nll(
            logits, example.input_ids, example.completion_mask, example.attention_mask
        )
        (loss * loss_scale).backward()
        if set(outputs) != set(names):
            raise ValueError("a projection was not executed")
        factors = {}
        for name in names:
            if outputs[name].grad is None:
                raise ValueError("missing projection derivative")
            derivative = outputs[name].grad[0].detach().float().cpu() / loss_scale
            if not torch.isfinite(activations[name]).all() or not torch.isfinite(derivative).all():
                raise FloatingPointError("nonfinite gradient factors")
            factors[name] = (activations[name], derivative)
        return factors, float(loss.detach())
    finally:
        for handle in handles:
            handle.remove()


def projection_scores(captures, *, incoming, outgoing):
    """Norm of the equal-example mean gradient for a full intermediate neuron.

    For G = B.T @ A, incoming row norms equal column norms of G.T. Reusing the
    tested Gram routine with transposed factors avoids full weight gradients.
    Incoming row and outgoing column squared norms add because they occupy
    disjoint parameter coordinates. Means are taken before norms, so opposing
    prompt gradients can cancel. All token positions contribute through the
    forward graph, including prompt positions that affect response predictions.
    """
    incoming = tuple(incoming)
    if not captures or not incoming or len(set((*incoming, outgoing))) != len(incoming) + 1:
        raise ValueError("nonempty captures and distinct incoming/outgoing projections required")
    incoming_norms = {}
    for name in incoming:
        incoming_norms[name] = gradient_statistics(
            [(row[name][1], row[name][0]) for row in captures]
        )["mean_gradient_norm"]
    outgoing_norm = gradient_statistics([row[outgoing] for row in captures])["mean_gradient_norm"]
    if any(norm.shape != outgoing_norm.shape for norm in incoming_norms.values()):
        raise ValueError("incoming rows and outgoing columns must share intermediate width")
    squared = outgoing_norm.square()
    for norm in incoming_norms.values():
        squared = squared + norm.square()
    return {
        "paired_gradient_norm": squared.sqrt(),
        "outgoing_gradient_norm": outgoing_norm,
        "incoming_gradient_norms": incoming_norms,
        "examples": len(captures),
    }
