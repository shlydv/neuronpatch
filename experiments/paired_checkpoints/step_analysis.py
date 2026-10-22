"""Separate the recorded Adam step into current-gradient and historical terms.

This is an offline arithmetic diagnosis, not an optimizer ablation. Both terms
use the same recorded second-moment denominator. Their directional derivatives
do not predict a large nonlinear model change or establish a momentum remedy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def decompose(state, audit):
    group = audit["optimizer_group"]
    assert group["weight_decay"] == 0
    beta1, beta2 = group["betas"]
    step = audit["step"]
    assert step > 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1
    assert group["lr"] > 0 and group["eps"] > 0
    names = audit["parameter_names"]
    assert names and len(names) == len(set(names))
    rows = []
    totals = {
        "actual_squared_norm": 0.0,
        "history_squared_norm": 0.0,
        "current_squared_norm": 0.0,
        "rounding_squared_norm": 0.0,
        "joint_squared_norm": 0.0,
        "correction_squared_norm": 0.0,
        "max_parameter_reconstruction_error": 0.0,
    }
    for gradient in ("joint", "correction"):
        for term in ("actual", "history", "current", "rounding"):
            totals[f"{gradient}_dot_{term}"] = 0.0
    for i, name in enumerate(names):
        before = state[f"parameter-before/{i}"].double()
        after = state[f"parameter-after/{i}"].double()
        joint = state[f"clipped-joint-gradient/{i}"].double()
        correction = state[f"correction-gradient/{i}"].double()
        m = state[f"adam-first-moment/{i}"].double()
        v = state[f"adam-second-moment/{i}"].double()
        assert all(t.shape == before.shape for t in (after, joint, correction, m, v))
        assert all(torch.isfinite(t).all() for t in (before, after, joint, correction, m, v))
        assert (v >= 0).all()
        denominator = ((beta2 * v + (1 - beta2) * joint.square()) / (1 - beta2**step)).sqrt()
        denominator = denominator + group["eps"]
        scale = -group["lr"] / (1 - beta1**step)
        history = scale * beta1 * m / denominator
        current = scale * (1 - beta1) * joint / denominator
        predicted = before + history + current
        torch.testing.assert_close(after, predicted, rtol=2e-6, atol=1e-8)
        actual = after - before
        rounding = actual - history - current
        row = {"parameter": name}
        for term, value in (
            ("actual", actual),
            ("history", history),
            ("current", current),
            ("rounding", rounding),
            ("joint", joint),
            ("correction", correction),
        ):
            key = f"{term}_squared_norm"
            row[key] = float(value.square().sum())
            totals[key] += row[key]
        for gradient, vector in (("joint", joint), ("correction", correction)):
            for term, value in (
                ("actual", actual),
                ("history", history),
                ("current", current),
                ("rounding", rounding),
            ):
                key = f"{gradient}_dot_{term}"
                row[key] = float((vector * value).sum())
                totals[key] += row[key]
        totals["max_parameter_reconstruction_error"] = max(
            totals["max_parameter_reconstruction_error"], float((after - predicted).abs().max())
        )
        rows.append(row)
    # For the joint gradient, its own contribution is a descent direction under
    # the fixed positive denominator, independent of the historical contribution.
    assert totals["joint_dot_current"] <= 0
    for gradient in ("joint", "correction"):
        predicted_dot = sum(
            totals[f"{gradient}_dot_{t}"] for t in ("history", "current", "rounding")
        )
        assert math.isclose(
            totals[f"{gradient}_dot_actual"], predicted_dot, abs_tol=1e-12, rel_tol=1e-10
        )
        denominator = math.sqrt(totals[f"{gradient}_squared_norm"] * totals["actual_squared_norm"])
        totals[f"{gradient}_cosine_actual"] = (
            totals[f"{gradient}_dot_actual"] / denominator if denominator else None
        )
    return {
        "step": step,
        "parameters": len(names),
        "totals": totals,
        "per_parameter": rows,
        "scope": (
            "Algebraic contributions to one recorded Adam update at the same denominator. "
            "Joint gradient is globally clipped; correction gradient is unclipped. "
            "Dots describe local autograd directions, not nonlinear counterfactual outcomes. "
            "The historical term includes earlier correction and auxiliary gradients. "
            "Removing it would not reproduce resetting Adam or retraining."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = "case-0-paired_mean-43-1-final-update"
    paths = {kind: args.results / f"{prefix}.{kind}" for kind in ("json", "safetensors")}
    audit = json.loads(paths["json"].read_text())
    result = decompose(load_file(str(paths["safetensors"])), audit)
    result["measured_correction_nll"] = {
        key: audit[key]
        for key in (
            "pre_update_grad_task_nll",
            "pre_update_no_grad_task_nll",
            "post_update_grad_task_nll",
            "post_update_no_grad_task_nll",
        )
    }
    result["input_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()
    }
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_parameter"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
