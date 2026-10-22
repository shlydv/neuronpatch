"""Backtrack real Adam updates against measured loss and anchored output drift.

The measurement must use an eval-mode model with a frozen, immutable base.
It must not mutate model buffers or non-optimizer parameters. This transaction
covers the optimizer's parameters, gradients, moments, learning rates and RNG;
it is not a transaction over arbitrary external state or a safety guarantee.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Measurement:
    task_nll: float
    mean_kl: float
    worst_kl: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("finite nonnegative loss and KL measurements required")


@dataclass(frozen=True)
class Budget:
    mean_kl: float
    worst_kl: float
    minimum_improvement: float = 1e-7

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("finite nonnegative preservation and improvement limits required")

    def accepts(self, before, after):
        return (
            after.task_nll <= before.task_nll - self.minimum_improvement
            and after.mean_kl <= self.mean_kl
            and after.worst_kl <= self.worst_kl
        )


def backtracked_step(optimizer, measure, before, budget, *, scales=(1.0, 0.5, 0.25, 0.125)):
    """Try a fixed-gradient Adam direction; commit one feasible update or restore all.

    Every retry starts from the same parameters and optimizer state. Rejected
    attempts therefore do not advance moments or step counters. The accepted
    retry retains exactly one scaled-rate Adam step, then restores the nominal
    learning rate for the next proposal. Exhaustion returns a no-change result.
    Standard Adam/AdamW without closures, hooks or differentiable stepping are
    supported in this experiment. Original model parameters stay outside Adam.
    """
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
        raise ValueError("standard Adam or AdamW required")
    scales = tuple(scales)
    if (
        not scales
        or any(not math.isfinite(s) or not 0 < s <= 1 for s in scales)
        or any(a <= b for a, b in zip(scales, scales[1:]))
    ):
        raise ValueError("strictly decreasing positive scales no greater than one required")
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("nonempty distinct optimizer parameters required")
    if any(
        p.grad is None or not torch.isfinite(p).all() or not torch.isfinite(p.grad).all()
        for p in parameters
    ):
        raise ValueError("finite parameters with finite existing gradients required")
    rates = [g["lr"] for g in optimizer.param_groups]
    if any(isinstance(r, torch.Tensor) or not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("positive finite scalar learning rates required")
    if any(g.get("differentiable", False) for g in optimizer.param_groups):
        raise ValueError("differentiable optimizer steps are unsupported")
    values = [p.detach().clone() for p in parameters]
    gradients = [p.grad.detach().clone() for p in parameters]
    state = deepcopy(optimizer.state_dict())
    cpu_rng = torch.random.get_rng_state()
    devices = sorted({p.device.index for p in parameters if p.device.type == "cuda"})
    cuda_rng = {d: torch.cuda.get_rng_state(d) for d in devices}

    def restore():
        with torch.no_grad():
            for p, value, grad in zip(parameters, values, gradients, strict=True):
                p.copy_(value)
                p.grad = grad.clone()
        # load_state_dict may reuse its input tensors; retain an immutable backup.
        optimizer.load_state_dict(deepcopy(state))
        torch.random.set_rng_state(cpu_rng)
        for d, rng in cuda_rng.items():
            torch.cuda.set_rng_state(rng, d)

    attempts = []
    try:
        for index, scale in enumerate(scales):
            if index:
                restore()
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * scale
            optimizer.step()
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("nonfinite proposed parameter")
            with torch.no_grad():
                measured = measure()
            if not isinstance(measured, Measurement):
                raise TypeError("measure must return a Measurement")
            accepted = budget.accepts(before, measured)
            attempts.append({"scale": scale, "measurement": asdict(measured), "accepted": accepted})
            if accepted:
                return {"accepted": True, "attempts": attempts, "measurement": measured}
        restore()
        return {"accepted": False, "attempts": attempts, "measurement": before}
    except BaseException:
        restore()
        raise
