"""Scoped finite-step checks; restore exact selected base slices on every exit."""

import math
from contextlib import contextmanager

import torch

from neuronpatch.paired import _layout


@contextmanager
def perturb(model, bank, selected, scale):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("eval-mode frozen base required")
    if (
        selected.dtype != torch.bool
        or selected.shape != bank.gain.shape
        or selected.device.type != "cpu"
        or not selected.any()
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("nonempty CPU candidate mask and finite positive scale required")
    checked = _layout(model, bank.selections)
    prepared = {}
    offset = 0
    with torch.no_grad():
        for selection in bank.selections:
            width = len(selection.indices)
            positions = selected[offset : offset + width].nonzero().flatten()
            offset += width
            if not len(positions):
                continue
            for name in (*selection.incoming, selection.outgoing):
                module, axis, indices = checked[name]
                dimension = 0 if axis == "output" else 1
                device_indices = torch.tensor(indices, device=module.weight.device)[
                    positions.to(module.weight.device)
                ]
                before = module.weight.index_select(dimension, device_indices).clone()
                direction = bank.matrices[name].index_select(dimension, positions)
                addition_dtype = torch.promote_types(before.dtype, torch.float32)
                update = (
                    before.to(addition_dtype) + scale * direction.to(before.device, addition_dtype)
                ).to(before.dtype)
                if not torch.isfinite(before).all() or not torch.isfinite(update).all():
                    raise ValueError("nonfinite or overflowing finite-step candidate")
                gradient = (
                    bank.gradients[name].index_select(dimension, positions)
                    if bank.gradients is not None
                    else None
                )
                prepared[name] = (
                    module,
                    dimension,
                    device_indices,
                    before,
                    update,
                    direction,
                    gradient,
                )
        applied = []
        try:
            for name, (module, dimension, indices, _, update, _, _) in prepared.items():
                applied.append(name)
                module.weight.index_copy_(dimension, indices, update)
            actual_squared = sum(
                float((row[4].double() - row[3].double()).square().sum())
                for row in prepared.values()
            )
            yield {
                "nominal_update_l2": scale * float(bank.squared_norm[selected].sum().sqrt()),
                "actual_quantized_update_l2": math.sqrt(actual_squared),
                "base": {name: row[3].cpu() for name, row in prepared.items()},
                "after": {name: row[4].cpu() for name, row in prepared.items()},
                "directions": {name: row[5] for name, row in prepared.items()},
                "gradients": {name: row[6] for name, row in prepared.items() if row[6] is not None},
            }
        finally:
            for name in applied:
                module, dimension, indices, before, _, _, _ = prepared[name]
                module.weight.index_copy_(dimension, indices, before)
            assert all(
                torch.equal(row[0].weight.index_select(row[1], row[2]), row[3])
                for row in prepared.values()
            )
