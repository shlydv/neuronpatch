"""Compact, mergeable updates to selected columns or rows of a linear projection."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def validate_indices(indices: Sequence[int], width: int) -> tuple[int, ...]:
    values = tuple(indices)
    if not values or any(type(i) is not int for i in values):
        raise ValueError("indices must be a non-empty sequence of Python integers")
    if len(set(values)) != len(values):
        raise ValueError("indices must be unique")
    if min(values) < 0 or max(values) >= width:
        raise ValueError(f"indices must be in [0, {width})")
    return values


class ColumnPatch(nn.Module):
    """Learn output vectors for selected input features of a frozen nn.Linear.

    For PyTorch's W[out, in], y = x W.T + b + x[..., indices] delta.T.
    In an MLP down-projection, these input features are intermediate neurons.
    The delta has shape [out_features, number_of_selected_neurons].
    """

    def __init__(self, base: nn.Linear, indices: Sequence[int]):
        super().__init__()
        if type(base) is not nn.Linear or not base.weight.is_floating_point():
            raise TypeError("only ordinary floating-point nn.Linear modules are supported")
        if base.weight.device.type == "meta":
            raise ValueError("materialize the model before attaching patches")
        values = validate_indices(indices, base.in_features)
        self.base = base
        self.base.requires_grad_(False)
        self.register_buffer("indices", torch.tensor(values, device=base.weight.device))
        self.delta = nn.Parameter(base.weight.new_zeros(base.out_features, len(values)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = x.index_select(-1, self.indices)
        return self.base(x) + F.linear(selected, self.delta)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return a standalone linear layer; do not mutate this patch or its base."""
        from copy import deepcopy

        merged = deepcopy(self.base)
        merged.weight.index_add_(1, self.indices, self.delta.to(merged.weight.dtype))
        return merged


class RowPatch(nn.Module):
    """Learn selected output rows: changes activation conditions in gate/up layers.

    Delta is [selected_outputs, in_features]. Biases remain frozen. Indices refer
    to output features, unlike ColumnPatch's input features.
    """

    def __init__(self, base: nn.Linear, indices: Sequence[int]):
        super().__init__()
        if type(base) is not nn.Linear or not base.weight.is_floating_point():
            raise TypeError("only ordinary floating-point nn.Linear modules are supported")
        if base.weight.device.type == "meta":
            raise ValueError("materialize the model before attaching patches")
        values = validate_indices(indices, base.out_features)
        self.base = base
        self.base.requires_grad_(False)
        self.register_buffer("indices", torch.tensor(values, device=base.weight.device))
        self.delta = nn.Parameter(base.weight.new_zeros(len(values), base.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x).index_add(-1, self.indices, F.linear(x, self.delta))

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        from copy import deepcopy

        merged = deepcopy(self.base)
        merged.weight.index_add_(0, self.indices, self.delta.to(merged.weight.dtype))
        return merged
