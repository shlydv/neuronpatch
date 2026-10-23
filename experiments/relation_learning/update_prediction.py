"""Prospective first-order loss checks using the optimizer's actual next update.

Experimental instrumentation, not a selector or a protection guarantee. Call
after gradients are unscaled/clipped, before the real optimizer step. Only compact
trainables and their optimizer state are copied; never pass full model weights.
The caller owns the deterministic calibration forward and the real step.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch


@dataclass
class Proposal:
    parameters: tuple
    before: tuple
    after: tuple

    @property
    def deltas(self):
        return tuple(after - before for before, after in zip(self.before, self.after, strict=True))

    def check_applied(self):
        """Fail if the forecasted update differs from the update actually applied."""
        if not all(
            torch.equal(p.detach(), after)
            for p, after in zip(self.parameters, self.after, strict=True)
        ):
            raise ValueError("Actual parameters differ from the forecasted optimizer step")


def preview(optimizer, *, max_scalars=2_000_000):
    """Preview ordinary Adam/AdamW with momentum, groups, decay and skipped grads.

    Execute PyTorch's optimizer on independent parameters/state instead of
    approximating its formula. Original parameters, gradients and moments are
    untouched. Optimizer hooks, differentiable/capturable execution and closure
    optimizers are outside this diagnostic's contract. Apply the real step with
    unchanged state/gradients, then call check_applied().
    """
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
        raise ValueError("Only ordinary Adam and AdamW are supported")
    if any(g.get("differentiable") or g.get("capturable") for g in optimizer.param_groups):
        raise ValueError("Differentiable/capturable optimizers are unsupported")
    parameters = tuple(p for g in optimizer.param_groups for p in g["params"])
    if (
        type(max_scalars) is not int
        or max_scalars <= 0
        or not parameters
        or len({id(p) for p in parameters}) != len(parameters)
        or sum(p.numel() for p in parameters) > max_scalars
    ):
        raise ValueError("Unique compact parameters within a positive scalar cap required")
    if any(
        not p.is_floating_point()
        or not torch.isfinite(p).all()
        or (p.grad is not None and (p.grad.is_sparse or not torch.isfinite(p.grad).all()))
        for p in parameters
    ):
        raise ValueError("Finite floating parameters and finite dense gradients required")
    before = tuple(p.detach().clone() for p in parameters)
    clones = []
    groups = []
    for group in optimizer.param_groups:
        copied = []
        for p in group["params"]:
            q = torch.nn.Parameter(p.detach().clone())
            q.grad = None if p.grad is None else p.grad.detach().clone()
            copied.append(q)
        clones.extend(copied)
        groups.append(
            {**{k: deepcopy(v) for k, v in group.items() if k != "params"}, "params": copied}
        )
    shadow = type(optimizer)(groups)
    shadow.load_state_dict(deepcopy(optimizer.state_dict()))
    shadow.step()
    after = tuple(p.detach().clone() for p in clones)
    if not all(torch.isfinite(p).all() for p in after):
        raise FloatingPointError("Nonfinite forecasted optimizer step")
    return Proposal(parameters, before, after)


def predict_loss(proposal, loss):
    """Predict loss change without accumulating into training .grad buffers.

    The provided scalar loss must be evaluated at the unchanged pre-step model
    on explicitly separate calibration examples. Negative change means improvement.
    This local prediction can fail at finite steps; measure after the real step.
    """
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Finite scalar calibration loss required")
    if not all(
        torch.equal(p.detach(), before)
        for p, before in zip(proposal.parameters, proposal.before, strict=True)
    ):
        raise ValueError("Prediction must be recorded before applying the proposed update")
    gradients = torch.autograd.grad(loss, proposal.parameters, allow_unused=True)
    predicted = 0.0
    for gradient, delta in zip(gradients, proposal.deltas, strict=True):
        if gradient is not None:
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite calibration derivative")
            predicted += float((gradient.detach().double() * delta.double()).sum())
    return {"before_loss": float(loss.detach()), "predicted_change": predicted}


def compare(proposal, prediction, after_loss):
    proposal.check_applied()
    after = float(after_loss.detach())
    if not math.isfinite(after):
        raise FloatingPointError("Nonfinite observed calibration loss")
    observed = after - prediction["before_loss"]
    return {
        **prediction,
        "after_loss": after,
        "observed_change": observed,
        "linearization_error": observed - prediction["predicted_change"],
        "same_sign": (observed > 0) - (observed < 0)
        == (prediction["predicted_change"] > 0) - (prediction["predicted_change"] < 0),
    }
