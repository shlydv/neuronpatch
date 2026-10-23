"""Experimental activation-probe selection and fixed two-cluster sharing.

Inspired by NeST, not its exact discovery recipe: fit a standardized logistic
probe with explicit L2 regularization, select a fixed absolute-coefficient budget,
and use two profile clusters. No causal interpretation or advantage is assumed.
Both guided and random supports use the same clustering procedure and row budget.
Only clean/injected TRAINING prompt activations may enter this diagnostic.
"""

import torch
from torch import nn
from torch.nn import functional as F


@torch.inference_mode()
def capture_prompt(model, inputs, names):
    """Max-pool projection OUTPUTS over nonpadding prompt tokens, one prompt.

    The caller supplies only prompt tokens, never teacher or evaluation answers.
    No model mode or parameters are changed; hooks are removed on all exits.
    """
    names = tuple(names)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Frozen eval-mode model required")
    mask = inputs["attention_mask"]
    if (
        not names
        or len(set(names)) != len(names)
        or mask.ndim != 2
        or mask.shape[0] != 1
        or not mask.bool().any()
        or inputs["input_ids"].shape != mask.shape
    ):
        raise ValueError("Unique projections and one nonempty masked prompt required")
    modules = {name: model.get_submodule(name) for name in names}
    if any(type(module) is not nn.Linear for module in modules.values()):
        raise ValueError("Unpatched ordinary Linear projections required")
    captured, handles = {}, []

    def hook(name):
        def record(module, args, output):
            if name in captured or output.ndim != 3 or output.shape[:2] != mask.shape:
                raise ValueError("Each projection must execute once over the full prompt")
            valid = output[0, mask[0].bool().to(output.device)]
            if not torch.isfinite(valid).all():
                raise FloatingPointError("Nonfinite selected prompt activations")
            captured[name] = valid.max(0).values.float().cpu()

        return record

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        model(**inputs, use_cache=False)
        if set(captured) != set(names):
            raise ValueError("A requested projection was not executed")
        return captured
    finally:
        for handle in handles:
            handle.remove()


def fit_probe(profiles, labels, *, max_iter=40):
    """Return coefficients in standardized coordinates and optimization diagnostics.

    Each row is one prompt, each column one neuron. Objective is mean binary
    cross-entropy + ||w||²/(2N); the intercept is not penalized. Constant columns
    become zero. A fit uses CPU FP64 without touching global RNG state.
    """
    x = profiles.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu", dtype=torch.float64)
    if (
        x.ndim != 2
        or not x.numel()
        or y.shape != (x.shape[0],)
        or not torch.isfinite(x).all()
        or set(y.tolist()) != {0.0, 1.0}
        or type(max_iter) is not int
        or max_iter <= 0
    ):
        raise ValueError(
            "Finite profiles, both binary classes, and positive iteration cap required"
        )
    mean, scale = x.mean(0), x.std(0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x = (x - mean) / scale
    weight = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    bias = torch.zeros((), dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=max_iter,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-10,
    )

    def objective():
        return F.binary_cross_entropy_with_logits(x @ weight + bias, y) + weight.square().sum() / (
            2 * len(y)
        )

    initial = float(objective().detach())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = objective()
    gradients = torch.autograd.grad(loss, (weight, bias))
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Nonfinite probe fit")
    return {
        "weight": weight.detach(),
        "mean": mean,
        "scale": scale,
        "bias": float(bias.detach()),
        "initial_objective": initial,
        "final_objective": float(loss.detach()),
        "gradient_max_abs": max(float(g.abs().max()) for g in gradients),
        "training_accuracy": float(((x @ weight + bias >= 0) == y.bool()).double().mean()),
    }


def two_clusters(profiles, *, max_iter=50):
    """Deterministic Lloyd clustering of neuron profiles across prompts.

    Rows are neurons. Farthest-point initialization avoids RNG changes. Identical
    profiles get a recorded deterministic split to maintain a two-parameter-group
    budget; this is not evidence of two natural clusters. Labels follow input order.
    """
    x = profiles.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not x.shape[1] or not torch.isfinite(x).all():
        raise ValueError("At least two finite nonempty neuron profiles required")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("Positive iteration cap required")
    first = int((x - x.mean(0)).square().sum(1).argmax())
    second = int((x - x[first]).square().sum(1).argmax())
    degenerate = bool(torch.equal(x[first], x[second]))
    centers = x[[first, second]].clone()
    previous = None
    for iteration in range(max_iter):
        distances = (x[:, None, :] - centers[None, :, :]).square().sum(-1)
        labels = distances.argmin(1)
        if len(labels.unique()) == 1:
            donor = int(distances[:, int(labels[0])].argmax())
            labels[donor] = 1 - labels[donor]
        if previous is not None and torch.equal(labels, previous):
            break
        centers = torch.stack([x[labels == label].mean(0) for label in (0, 1)])
        previous = labels.clone()
    return {
        "groups": labels.tolist(),
        "degenerate": degenerate,
        "iterations": iteration + 1,
        "inertia": float((x - centers[labels]).square().sum()),
    }


def select(profiles, probe, *, count, random_seed=None):
    """Select fixed-budget rows, then cluster standardized activation profiles."""
    if type(count) is not int or not 2 <= count <= profiles.shape[1]:
        raise ValueError("A neuron budget of2 through projection width is required")
    if random_seed is None:
        indices = torch.argsort(probe["weight"].abs(), descending=True, stable=True)[:count]
    else:
        indices = torch.randperm(
            profiles.shape[1], generator=torch.Generator().manual_seed(random_seed)
        )[:count]
    standardized = (profiles.detach().double().cpu() - probe["mean"]) / probe["scale"]
    result = two_clusters(standardized[:, indices].T)
    return {"indices": indices.tolist(), **result}
