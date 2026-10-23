"""Tiny algebra demonstration; no model downloads, training or efficacy claim."""

import json
from pathlib import Path

import torch


def main():
    torch.manual_seed(23)
    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    gate_weight = torch.randn(2, 3, dtype=torch.float64)
    up = torch.randn(2, 3, dtype=torch.float64)
    down = torch.randn(2, 2, dtype=torch.float64)

    def measure(g, u, d):
        g, u, d = [v.detach().clone().requires_grad_() for v in (g, u, d)]
        gate = torch.ones(2, dtype=torch.float64, requires_grad=True)
        y = ((torch.nn.functional.silu(x @ g.T) * (x @ u.T)) * gate) @ d.T
        loss = (y - target).square().mean()
        gg, gu, gd, gz = torch.autograd.grad(loss, (g, u, d, gate))
        score = gg.square().sum(1) + gu.square().sum(1) + gd.square().sum(0)
        return y.detach(), score.detach(), gz.detach()

    y, a, z = measure(gate_weight, up, down)
    scale = torch.ones(2, dtype=torch.float64)
    scale[int(a.argmin())] = 1000
    other, b, zz = measure(gate_weight, up * scale[:, None], down / scale[None, :])
    torch.testing.assert_close(y, other, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(z, zz, rtol=1e-12, atol=1e-12)
    assert int(a.argmax()) != int(b.argmax())
    record = {
        "scope": "Tiny algebra check, not an LLM efficacy experiment",
        "seed": 23,
        "rescaling": scale.tolist(),
        "output_max_abs_difference": float((y - other).abs().max()),
        "raw_group_gradient_squared_before": a.tolist(),
        "raw_group_gradient_squared_after": b.tolist(),
        "top_neuron_before": int(a.argmax()),
        "top_neuron_after": int(b.argmax()),
        "gate_gradient_before": z.tolist(),
        "gate_gradient_after": zz.tolist(),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
