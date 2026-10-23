"""Experimental shared row updates; selection and grouping are supplied explicitly.

Implements the shared additive update in NeST equation 13 using PyTorch's
weight[out, in] convention. This is not the paper's discovery/training pipeline.
The high-level NeuronPatch API stores explicit group assignments with the deltas.
"""

from collections.abc import Sequence
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .layers import validate_indices


def validate_groups(groups: Sequence[int], count: int) -> tuple[int, ...]:
    assignments = tuple(groups)
    if len(assignments) != count or any(type(g) is not int for g in assignments):
        raise ValueError("groups must provide one Python integer per selected row")
    unique = set(assignments)
    if not unique or unique != set(range(len(unique))):
        raise ValueError("group labels must be contiguous starting at zero")
    return assignments


class GroupedRowPatch(nn.Module):
    """Tie selected output-row deltas according to an explicit group assignment.

    `groups[j]` assigns `indices[j]` to one of 0..n_groups-1. Delta has shape
    [n_groups, in_features]. Group gradients sum their member-row contributions;
    they are not averaged. Large groups therefore change optimization geometry.
    No bias or base weight is trainable. Merge removes the extra forward work.
    Explicit delta_dtype keeps updates in FP32/FP64 on a lower-precision base;
    omitting it preserves the base-dtype forward and merge semantics.
    """

    def __init__(
        self,
        base: nn.Linear,
        indices: Sequence[int],
        groups: Sequence[int],
        *,
        delta_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if delta_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("explicit delta_dtype must be float32 or float64")
        if type(base) is not nn.Linear or not base.weight.is_floating_point():
            raise TypeError("only ordinary floating-point nn.Linear modules are supported")
        if base.weight.device.type == "meta":
            raise ValueError("materialize the model before attaching patches")
        values = validate_indices(indices, base.out_features)
        assignments = validate_groups(groups, len(values))
        unique = set(assignments)
        # Validate first so invalid requests leave the supplied layer untouched.
        self.base = base.requires_grad_(False)
        self.register_buffer("indices", torch.tensor(values, device=base.weight.device))
        self.register_buffer("groups", torch.tensor(assignments, device=base.weight.device))
        self.delta = nn.Parameter(
            torch.zeros(
                len(unique),
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype if delta_dtype is None else delta_dtype,
            )
        )
        self.explicit_delta_dtype = delta_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.explicit_delta_dtype is None:
            updates = F.linear(x, self.delta)
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                updates = F.linear(x.to(self.delta.dtype), self.delta)
        return output.index_add(
            -1, self.indices, updates.index_select(-1, self.groups).to(output.dtype)
        )

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return an ordinary independent linear layer, leaving the base unchanged."""
        merged = deepcopy(self.base)
        updates = self.delta.index_select(0, self.groups)
        if self.explicit_delta_dtype is None:
            merged.weight.index_add_(0, self.indices, updates.to(merged.weight.dtype))
        else:
            dtype = torch.promote_types(merged.weight.dtype, updates.dtype)
            selected = merged.weight.index_select(0, self.indices).to(dtype)
            merged.weight.index_copy_(
                0, self.indices, (selected + updates.to(dtype)).to(merged.weight.dtype)
            )
        return merged
