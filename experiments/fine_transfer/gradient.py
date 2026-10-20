"""Experimental cross-prompt selection by agreement of actual update gradients.

This is not integrated gradients, a FiNE reproduction, or a novelty claim.
For a linear projection and one scalar loss per prompt, the column gradient is
G_j = B.T @ A[:, j], including ALL token positions. Agreement scores estimate
the worst first-order loss reduction along the normalized mean column gradient.
They do not guarantee transfer, utility preservation or Adam update behavior.
"""

import torch
from torch.nn import functional as F


def agreement_scores(factors):
    """Return per-column worst-prompt gains without materializing weight gradients.

    Each (A, B) contains [tokens, input_width] activations and [tokens, output_width]
    output-loss derivatives for one projection and prompt. Gram products compute
    inner products between the full column gradients. Float64 CPU arithmetic
    avoids retaining dense model-sized gradients or GPU tensors.
    """
    if not factors:
        raise ValueError("At least one prompt required")
    a, b, sizes = [], [], []
    for inputs, derivatives in factors:
        if (
            inputs.ndim != 2
            or derivatives.ndim != 2
            or inputs.shape[0] != derivatives.shape[0]
            or min(*inputs.shape, *derivatives.shape) <= 0
        ):
            raise ValueError("Matching nonempty token matrices required")
        a.append(inputs.detach().double().cpu())
        b.append(derivatives.detach().double().cpu())
        sizes.append(len(inputs))
    if len({x.shape[1] for x in a}) != 1 or len({x.shape[1] for x in b}) != 1:
        raise ValueError("Widths must agree across prompts")
    if not all(torch.isfinite(x).all() for x in a + b):
        raise ValueError("Finite factors required")
    activations, derivatives = torch.cat(a), torch.cat(b)
    projected = (derivatives @ derivatives.T) @ activations / len(factors)
    dots, start = [], 0
    for count in sizes:
        end = start + count
        dots.append((activations[start:end] * projected[start:end]).sum(0))
        start = end
    dots = torch.stack(dots)
    norm_squared = dots.mean(0).clamp_min(0)
    norms = norm_squared.sqrt()
    gains = torch.zeros_like(norms)
    nonzero = norms > 0
    gains[nonzero] = dots[:, nonzero].min(0).values / norms[nonzero]
    return gains


def capture_gradient_factors(
    model, input_ids, target_position, target_token, names, loss_scale=128
):
    """Capture exact local factors for one teacher-forced target-token CE.

    Frozen embedding outputs are made differentiable so reverse-mode derivatives
    traverse the frozen model. No base parameter requires gradients. Hooks are
    always removed; callers must supply an eval-mode, unpatched model.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Use an eval-mode, frozen base")
    activations, outputs, handles = {}, {}, []

    def hook(name):
        def collect(module, args, output):
            if name in outputs:
                raise ValueError("Projection executed more than once")
            activations[name] = args[0][0].detach().float().cpu()
            output.retain_grad()
            outputs[name] = output

        return collect

    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
        embedded = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        logits = model(
            inputs_embeds=embedded, attention_mask=torch.ones_like(input_ids), use_cache=False
        ).logits
        loss = F.cross_entropy(
            logits[0, target_position].float().unsqueeze(0),
            torch.tensor([target_token], device=logits.device),
        )
        # Constant loss scaling limits FP16 underflow in frozen backward paths.
        # Returned derivatives are unscaled before computing agreement scores.
        (loss * loss_scale).backward()
        if set(outputs) != set(names):
            raise ValueError("A projection was not executed")
        result = {}
        for name in names:
            derivative = outputs[name].grad
            if derivative is None:
                raise ValueError("Missing projection derivative")
            result[name] = (activations[name], derivative[0].detach().float().cpu() / loss_scale)
        return result, float(loss.detach())
    finally:
        for handle in handles:
            handle.remove()
