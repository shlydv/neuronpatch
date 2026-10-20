"""Same-span null-basis LoRA control, outside the public library API.

Independently implements B C N.T from arXiv:2609.25618, Section 3.4. Supplied
spans use our exact numerical-rank rule, not the paper's 99%-energy cutoff.
This is a matched protection control, not a full paper reproduction.
"""

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .prototype import InputSpan


@torch.no_grad()
def null_basis(span):
    """Complete an already fitted protected basis without changing its rank."""
    if not isinstance(span, InputSpan) or span.basis.ndim != 2 or span.free_directions <= 0:
        raise ValueError("A non-full protected input span is required")
    protected = span.basis.detach().double()
    if not torch.isfinite(protected).all():
        raise ValueError("Protected basis must be finite")
    if not torch.allclose(
        protected.T @ protected,
        torch.eye(span.rank, dtype=protected.dtype, device=protected.device),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Protected basis must be orthonormal")
    if span.rank == 0:
        return torch.eye(protected.shape[0], dtype=protected.dtype, device=protected.device)
    complete, _ = torch.linalg.qr(protected, mode="complete")
    return complete[:, span.rank :].contiguous()


class NullBasisLoRA(nn.Module):
    """Learn low-rank updates in the same allowed input space as protected rows.

    All output rows are available. C has shape [rank, free_input_directions];
    B has shape [output_features, rank]. Projection bases and the original model
    are frozen. Computation uses FP64, matching the experimental row adapter.
    Rank/alpha, target layers, data and tuning budget must be controlled by the
    caller. No held-out preservation or quality guarantee follows from this class.
    """

    def __init__(self, model, spans, *, rank, alpha):
        super().__init__()
        if not spans or not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("Nonempty target spans and positive integer rank are required")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Alpha must be finite and positive")
        modules = list(model.named_modules(remove_duplicate=False))
        parameters = list(model.named_parameters(remove_duplicate=False))
        checked = []
        for name, span in spans.items():
            module = model.get_submodule(name)
            if (
                type(module) is not nn.Linear
                or module.weight.device.type == "meta"
                or not module.weight.is_floating_point()
            ):
                raise ValueError("Ordinary materialized floating-point projections are required")
            if sum(item is module for _, item in modules) != 1 or any(
                sum(item is parameter for _, item in parameters) != 1
                for parameter in module.parameters()
            ):
                raise ValueError("Shared projections or tied target parameters are unsupported")
            basis = null_basis(span)
            if basis.shape[0] != module.in_features:
                raise ValueError("Span width differs from the projection input width")
            if rank > min(module.out_features, basis.shape[1]):
                raise ValueError("Rank exceeds the available input/output dimensions")
            checked.append((name, module, basis))
        self.model = model
        self.rank = rank
        self.scale = alpha / rank
        self.left = nn.ParameterList()
        self.right = nn.ParameterList()
        self.names = []
        self._active = False
        self._exported = False
        for index, (name, module, basis) in enumerate(checked):
            device = module.weight.device
            self.register_buffer(f"null_{index}", basis.to(device))
            self.left.append(
                nn.Parameter(
                    torch.zeros(module.out_features, rank, device=device, dtype=torch.float64)
                )
            )
            right = torch.empty(rank, basis.shape[1], device=device, dtype=torch.float64)
            nn.init.kaiming_uniform_(right, a=math.sqrt(5))
            self.right.append(nn.Parameter(right))
            self.names.append(name)
        # Freeze only after every supplied target passes validation/allocation.
        self.model.requires_grad_(False)

    def trainable_parameters(self):
        yield from self.left
        yield from self.right

    @property
    def num_trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def effective_weight(self, index):
        return self.scale * self.left[index] @ self.right[index] @ getattr(self, f"null_{index}").T

    @contextmanager
    def applied(self):
        if self._active or self._exported:
            raise RuntimeError("Adapter is already active or was exported")
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("Use eval mode to match calibrated inputs")
        handles = []
        self._active = True
        try:
            for index, name in enumerate(self.names):

                def hook(module, inputs, output, index=index):
                    hidden = inputs[0].double() @ getattr(self, f"null_{index}")
                    hidden = F.linear(hidden, self.right[index])
                    update = self.scale * F.linear(hidden, self.left[index])
                    return output + update.to(output.dtype)

                handles.append(self.model.get_submodule(name).register_forward_hook(hook))
            yield self.model
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    def forward(self, *args, **kwargs):
        with self.applied():
            return self.model(*args, **kwargs)

    @torch.no_grad()
    def export_factors(self):
        """Return ordinary A/B LoRA factors in base dtype, with scale folded into B.

        The input factor expands C N.T once, so deployment need not carry the
        usually larger null basis. These tensors are not a PEFT artifact format.
        Low-precision factor rounding and merge must be evaluated separately.
        Export invalidates the training wrapper but does not change base weights.
        """
        if self._active or self._exported:
            raise RuntimeError("Cannot export an active or previously exported adapter")
        factors = {}
        for index, name in enumerate(self.names):
            dtype = self.model.get_submodule(name).weight.dtype
            a = (self.right[index] @ getattr(self, f"null_{index}").T).to(dtype)
            b = (self.scale * self.left[index]).to(dtype)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise FloatingPointError("Nonfinite deployment factor")
            factors[name] = {"a": a, "b": b}
        self._exported = True
        return factors
