"""Experimental incoming-row updates with a fixed protected input span.

Outside the public API pending actual model-quality evidence. The guarantee is
linear-algebraic on calibrated input vectors in exact arithmetic. It does not
guarantee preservation on new prompts or after low-precision export.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from neuronpatch import NeuronPatch
from neuronpatch.layers import validate_indices


@dataclass(frozen=True)
class InputSpan:
    basis: torch.Tensor
    samples: int
    relative_tolerance: float
    max_residual: float

    @property
    def rank(self):
        return self.basis.shape[1]

    @property
    def free_directions(self):
        return self.basis.shape[0] - self.rank


@torch.no_grad()
def protected_span(inputs: torch.Tensor) -> InputSpan:
    """Capture the numerical row span of all supplied, unpadded input vectors.

    Uses FP64 SVD and the usual size-scaled machine-precision rank tolerance.
    No statistical rank truncation or preservation relaxation is performed.
    Callers choose calibration data and explicitly exclude padding. A full input
    span has no protected adaptation direction and is rejected.
    """
    if (
        inputs.ndim != 2
        or not inputs.is_floating_point()
        or min(inputs.shape) == 0
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError("finite nonempty [tokens, input_features] inputs required")
    values = inputs.detach().double()
    # CUDA's QR-based driver handles nearly dependent calibration vectors robustly.
    _, singular, vh = torch.linalg.svd(
        values, full_matrices=False, driver="gesvd" if values.is_cuda else None
    )
    rtol = max(values.shape) * torch.finfo(values.dtype).eps
    rank = int((singular > singular.max() * rtol).sum())
    if rank == values.shape[1]:
        raise ValueError("protected inputs span the entire input space; no editing capacity")
    basis = vh[:rank].T.contiguous()
    residual = values - (values @ basis) @ basis.T
    return InputSpan(basis, values.shape[0], rtol, float(residual.abs().max()))


class ProtectedRowEdit(nn.Module):
    """Train rows D through D_eff = D - (D Q) Q.T on an unchanged eval-mode base.

    Q is a fixed orthonormal basis of protected input vectors. The raw trainable
    rows have redundant coordinates; allocated parameter count and effective
    linear dimension are reported separately. FP64 update arithmetic limits
    projection leakage; updates are cast to the base output dtype before addition.
    """

    def __init__(self, model, targets, spans):
        super().__init__()
        if not targets or set(targets) != set(spans):
            raise ValueError("targets and spans must have the same nonempty keys")
        modules = list(model.named_modules(remove_duplicate=False))
        parameters = list(model.named_parameters(remove_duplicate=False))
        checked = []
        for name, rows in targets.items():
            module = model.get_submodule(name)
            if type(module) is not nn.Linear or module.weight.device.type == "meta":
                raise ValueError("ordinary materialized projections required")
            if not module.weight.is_floating_point():
                raise ValueError("floating-point projections required")
            if sum(m is module for _, m in modules) != 1 or any(
                sum(p is parameter for _, p in parameters) != 1 for parameter in module.parameters()
            ):
                raise ValueError("shared projections or tied target parameters are unsupported")
            indices = validate_indices(rows, module.out_features)
            span = spans[name]
            if (
                not isinstance(span, InputSpan)
                or span.basis.ndim != 2
                or span.basis.shape[0] != module.in_features
                or span.free_directions <= 0
                or not span.basis.is_floating_point()
                or span.basis.device.type == "meta"
                or not torch.isfinite(span.basis).all()
            ):
                raise ValueError("a valid non-full protected input basis is required")
            basis = span.basis.detach().double()
            if not torch.allclose(
                basis.T @ basis,
                torch.eye(span.rank, dtype=basis.dtype, device=basis.device),
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError("the protected basis must be orthonormal")
            checked.append((name, module, indices, basis))
        self.model = model
        self.targets = {name: tuple(indices) for name, _, indices, _ in checked}
        self.rows = nn.ParameterList()
        self._names = []
        self._active = False
        self._exported = False
        self.effective_dimensions = 0
        for index, (name, module, indices, basis) in enumerate(checked):
            device = module.weight.device
            self.register_buffer(f"basis_{index}", basis.clone().to(device))
            self.register_buffer(f"indices_{index}", torch.tensor(indices, device=device))
            self.rows.append(
                nn.Parameter(
                    torch.zeros(
                        len(indices),
                        module.in_features,
                        device=device,
                        dtype=torch.float64,
                    )
                )
            )
            self._names.append(name)
            self.effective_dimensions += len(indices) * (module.in_features - basis.shape[1])
        # Complete validation/allocation before freezing the supplied model.
        self.model.requires_grad_(False)

    def trainable_parameters(self):
        yield from self.rows

    @property
    def num_trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.rows)

    def effective_weight(self, index):
        raw = self.rows[index]
        basis = getattr(self, f"basis_{index}")
        return raw - (raw @ basis) @ basis.T

    @contextmanager
    def applied(self):
        if self._active or self._exported:
            raise RuntimeError("edit is already active or was exported")
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("use eval mode to match calibrated input patterns")
        self._active = True
        handles = []
        try:
            for index, name in enumerate(self._names):

                def hook(module, inputs, output, index=index):
                    update = F.linear(inputs[0].double(), self.effective_weight(index))
                    return output.index_add(
                        -1, getattr(self, f"indices_{index}"), update.to(output.dtype)
                    )

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
    def export(self):
        """Compile projected rows into the ordinary patch; invalidate this editor.

        Uses the base dtype for the saved row deltas, so evaluate rounding effects
        separately. The training basis/optimizer are not part of deployment.
        """
        if self._active or self._exported:
            raise RuntimeError("cannot export an active or previously exported editor")
        updates = {name: self.effective_weight(i).detach() for i, name in enumerate(self._names)}
        if any(not torch.isfinite(update).all() for update in updates.values()):
            raise ValueError("cannot export nonfinite row updates")
        patch = NeuronPatch(self.model, self.targets, axis="output")
        for name, update in updates.items():
            patch.layers[name].delta.copy_(update.to(patch.layers[name].delta.dtype))
        self._exported = True
        return patch
