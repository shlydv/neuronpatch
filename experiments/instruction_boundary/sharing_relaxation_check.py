"""CPU algebra/optimizer check for relaxing fixed row sharing into masked LoRA.

This is a diagnostic fixture, not a model training implementation or evidence
of behavioral improvement. No base model or saved experiment answers are used.
"""

import json

import torch
from torch.nn import functional as F

from neuronpatch.grouped import GroupedRowPatch


def check():
    generator = torch.Generator().manual_seed(430)
    base = torch.nn.Linear(7, 11, bias=False, dtype=torch.float64)
    with torch.no_grad():
        base.weight.copy_(torch.randn(11, 7, generator=generator, dtype=torch.float64))
    rows, labels = [8, 1, 5, 3], [0, 1, 0, 1]
    grouped = GroupedRowPatch(base, rows, labels, delta_dtype=torch.float64)
    basis = torch.nn.Parameter(torch.zeros_like(grouped.delta))
    mixing = torch.nn.Parameter(F.one_hot(torch.tensor(labels), 2).double())
    initial_mixing = mixing.detach().clone()
    x = torch.randn(6, 7, generator=generator, dtype=torch.float64)
    target = torch.randn(6, 11, generator=generator, dtype=torch.float64)

    def relaxed_output():
        selected = F.linear(F.linear(x, basis), mixing)
        return base(x).index_add(-1, grouped.indices, selected)

    fixed_optimizer = torch.optim.Adam([grouped.delta], lr=0.03)
    relaxed_optimizer = torch.optim.Adam([basis, mixing], lr=0.03)
    F.mse_loss(grouped(x), target).backward()
    F.mse_loss(relaxed_output(), target).backward()
    initial_basis_gradient_error = float((grouped.delta.grad - basis.grad).abs().max())
    initial_mixing_gradient = float(mixing.grad.abs().max())
    assert initial_basis_gradient_error < 1e-12
    assert initial_mixing_gradient == 0
    fixed_optimizer.step()
    relaxed_optimizer.step()
    first_step_output_error = float((grouped(x) - relaxed_output()).abs().max().detach())
    assert first_step_output_error < 1e-12
    relaxed_optimizer.zero_grad()
    F.mse_loss(relaxed_output(), target).backward()
    second_mixing_gradient = float(mixing.grad.abs().max())
    assert second_mixing_gradient > 0
    relaxed_optimizer.step()
    mixing_change = float((mixing - initial_mixing).abs().max().detach())
    assert mixing_change > 0

    # Membership in the larger function family is exact at arbitrary deltas.
    arbitrary_basis = torch.randn(2, 7, generator=generator, dtype=torch.float64)
    assert torch.equal(initial_mixing @ arbitrary_basis, arbitrary_basis[labels])
    # Same-group rows may acquire different coefficients after relaxation.
    alternative_mixing = initial_mixing.clone()
    alternative_mixing[2] = -alternative_mixing[0]
    alternative_rows = alternative_mixing @ arbitrary_basis
    assert not torch.equal(alternative_rows[0], alternative_rows[2])
    return {
        "scope": "Tiny CPU algebra and Adam check; no behavioral or novelty claim",
        "initial_basis_gradient_max_error": initial_basis_gradient_error,
        "initial_mixing_gradient_max": initial_mixing_gradient,
        "first_adam_step_output_max_error": first_step_output_error,
        "second_step_mixing_gradient_max": second_mixing_gradient,
        "second_step_mixing_parameter_max_change": mixing_change,
        "qwen4b_grouped_scalars": 72 * 2 * 2560,
        "qwen4b_relaxed_scalars": 72 * (2 * 2560 + 32 * 2),
        "extra_scalar_fraction": 32 / 2560,
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
