"""Score initial candidate update directions against reference-output curvature.

Only shortlisted neuron blocks are materialized on CPU. This is neither a full
Fisher inverse nor a promise that Adam keeps its initial direction during fitting.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class DirectionBank:
    selections: tuple
    matrices: dict
    gain: torch.Tensor
    squared_norm: torch.Tensor
    kind: str
    gradients: dict | None = None


def _gradient(captures, name):
    result = None
    for capture in captures:
        inputs, derivative = capture[name]
        if inputs.ndim != 2 or derivative.ndim != 2 or len(inputs) != len(derivative):
            raise ValueError("two-dimensional equal-token gradient factors required")
        value = derivative.double().T @ inputs.double()
        result = value if result is None else result + value
    return result / len(captures)


def build_directions(captures, selections, *, kind="adam_first", epsilon=1e-8):
    """Unit-normalize each full paired-neuron proposal under a fresh optimizer.

    `gradient` uses -g. `adam_first` uses -g/(|g|+epsilon), the first-step
    direction of fresh,unclipped Adam with zero weight decay and unit rate.
    g is the equal-example mean correction gradient. Bias corrections cancel
    in this first step. Actual later updates and global gradient clipping are
    not modeled. Matrices retain candidate order from restrict_factors().
    """
    selections = tuple(selections)
    names = [n for s in selections for n in (*s.incoming, s.outgoing)]
    if (
        not captures
        or not selections
        or len(names) != len(set(names))
        or kind not in ("gradient", "adam_first")
        or isinstance(epsilon, bool)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("correction factors,disjoint selections and explicit direction required")
    matrices, all_gradients, gains, norms = {}, {}, [], []
    for selection in selections:
        local = {
            name: _gradient(captures, name) for name in (*selection.incoming, selection.outgoing)
        }
        all_gradients.update(local)
        outgoing = local[selection.outgoing]
        width = len(selection.indices)
        if (
            outgoing.ndim != 2
            or outgoing.shape[1] != width
            or any(local[name].shape != (width, outgoing.shape[0]) for name in selection.incoming)
        ):
            raise ValueError("restricted incoming rows and outgoing columns must share layout")
        proposed = {
            name: -g if kind == "gradient" else -g / (g.abs() + epsilon)
            for name, g in local.items()
        }
        squared = sum(
            d.square().sum(1 if name in selection.incoming else 0) for name, d in proposed.items()
        )
        scale = squared.sqrt().clamp_min(torch.finfo(torch.float64).tiny)
        gain = torch.zeros(width, dtype=torch.float64)
        norm = torch.zeros_like(gain)
        for name, d in proposed.items():
            axis = 1 if name in selection.incoming else 0
            d = d / (scale[:, None] if axis == 1 else scale[None, :])
            matrices[name] = d
            gain -= (local[name] * d).sum(axis)
            norm += d.square().sum(axis)
        if not torch.isfinite(gain).all() or not torch.isfinite(norm).all():
            raise FloatingPointError("nonfinite proposed directions")
        gains.append(gain)
        norms.append(norm)
    return DirectionBank(
        selections, matrices, torch.cat(gains), torch.cat(norms), kind, all_gradients
    )


def probe_dots(bank, factors):
    """Exact per-neuron dot of the proposal with a factorized probe VJP.

    Contributions from all coupled projections add before squaring, retaining
    their cross terms. Summing these dots across a selected support similarly
    measures its joint sketch response,including between-neuron cross terms.
    """
    result = []
    for selection in bank.selections:
        value = torch.zeros(len(selection.indices), dtype=torch.float64)
        for name in selection.incoming:
            inputs, derivative = factors[name]
            direction = bank.matrices[name]
            value += ((inputs.double() @ direction.T) * derivative.double()).sum(0)
        inputs, derivative = factors[selection.outgoing]
        direction = bank.matrices[selection.outgoing]
        value += ((derivative.double() @ direction) * inputs.double()).sum(0)
        result.append(value)
    values = torch.cat(result)
    if values.shape != bank.gain.shape or not torch.isfinite(values).all():
        raise ValueError("finite probe factors must match the proposal layout")
    return values


def rank_scores(bank, responses, *, damping_ratio=1.0):
    """A per-group local gain/risk proxy,with measured joint risk kept separate.

    For a fixed direction d_i, linear gain is a_i=-g_i.d_i and preservation
    curvature is r_i=E[(v.d_i)^2]. Maximizing a_i*alpha minus
    (r_i+lambda*||d_i||²)*alpha²/2 gives a score proportional to
    a_i²/(r_i+lambda*||d_i||²). Lambda is a fixed multiple of mean curvature
    per unit direction norm over this shortlist,not tuned on evaluation outputs.
    The local model is a ranking hypothesis; it does not constrain later Adam.
    """
    if (
        responses.ndim != 2
        or len(responses) == 0
        or responses.shape[1:] != bank.gain.shape
        or not torch.isfinite(responses).all()
        or isinstance(damping_ratio, bool)
        or not math.isfinite(damping_ratio)
        or damping_ratio <= 0
    ):
        raise ValueError(
            "finite [balanced probes,candidates] responses and positive damping required"
        )
    risk = responses.double().square().mean(0)
    norm = bank.squared_norm
    informative = bool(risk.sum() > 0)
    damping = float(damping_ratio * risk.sum() / norm.sum()) if norm.sum() > 0 else 0.0
    denominator = risk + damping * norm if informative else norm
    scores = torch.zeros_like(bank.gain)
    valid = denominator > 0
    scores[valid] = bank.gain[valid].square() / denominator[valid]
    return {
        "scores": scores,
        "curvature": risk,
        "gain": bank.gain,
        "direction_squared_norm": norm,
        "damping": damping,
        "risk_informative": informative,
        "joint_curvature_all_candidates": float(responses.double().sum(1).square().mean()),
        "diagonal_curvature_all_candidates": float(risk.sum()),
    }
