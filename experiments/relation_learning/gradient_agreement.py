"""Diagnostic geometry for fixed candidate neurons, not a selection algorithm.

Use training/calibration examples only. Preserve signed compact gradients rather
than only Fisher magnitudes. This measures local loss geometry, not answer quality
or causality; actual finite updates must still be evaluated with update_prediction.
"""

import torch

from neuronpatch.grouped import validate_groups


def shared_gradient_geometry(gradients, groups):
    """Measure a fixed row-sharing constraint in effective-weight coordinates.

    If delta W = S U, the trainable gradient is S.T G (a sum). The orthogonal
    projection of G into that update space is S (S.T S)^-1 S.T G (group means).
    Confusing these two makes large groups look artificially stronger. Retained
    energy compares spaces under equal effective-weight Frobenius norm, not Adam
    steps, training outcomes, or cross-example generalization.
    """
    if gradients.ndim != 2 or not gradients.numel() or not torch.isfinite(gradients).all():
        raise ValueError("Finite nonempty row gradients required")
    groups = validate_groups(groups, gradients.shape[0])
    values = gradients.detach().to(device="cpu", dtype=torch.float64)
    labels = torch.tensor(groups)
    counts = torch.bincount(labels)
    summed = values.new_zeros((len(counts), values.shape[1])).index_add_(0, labels, values)
    projected = (summed / counts[:, None]).index_select(0, labels)
    total_energy = float(values.square().sum())
    return {
        "parameter_gradient": summed,
        "projected_weight_gradient": projected,
        "retained_energy_fraction": (
            float(projected.square().sum()) / total_energy if total_energy > 0 else None
        ),
        "group_sizes": counts.tolist(),
    }


def compact_gradients(capture, *, incoming, outgoing, indices):
    """Reconstruct selected incoming rows and outgoing columns, never full weights.

    Returned rows have a common neuron ordering and concatenate disjoint parameter
    coordinates. They match an independently trainable paired-neuron patch, not
    a shared cluster patch (whose parameter gradients sum member contributions).
    """
    incoming = tuple(incoming)
    indices = tuple(indices)
    names = (*incoming, outgoing)
    if not incoming or len(set(names)) != len(names):
        raise ValueError("Distinct incoming and outgoing projection names required")
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or i < 0 for i in indices)
    ):
        raise ValueError("Unique nonnegative integer neuron indices required")
    parts = []
    for name in names:
        a, b = capture[name]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
            raise ValueError("Matching token axes required")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Finite gradient factors required")
        a, b = (
            a.detach().to(device="cpu", dtype=torch.float64),
            b.detach().to(device="cpu", dtype=torch.float64),
        )
        width = a.shape[1] if name == outgoing else b.shape[1]
        if max(indices) >= width:
            raise ValueError("Neuron index outside projection width")
        if name == outgoing:
            parts.append(a[:, indices].T @ b)
        else:
            parts.append(b[:, indices].T @ a)
    return torch.cat(parts, dim=1)


class GradientBank:
    """Equal-example signed means and squared norms with bounded storage.

    One compact mean matrix is retained; example tensors are not retained.
    Caller must use the same fixed neuron and parameter-coordinate ordering in
    every bank. Repeated IDs fail to prevent silently overweighting an example.
    """

    def __init__(self, *, max_scalars=2_000_000):
        if type(max_scalars) is not int or max_scalars <= 0:
            raise ValueError("Positive integer storage cap required")
        self.max_scalars = max_scalars
        self.ids = set()
        self.mean = None
        self.mean_squared_norm = None

    def add(self, example_id, gradient):
        if not isinstance(example_id, str) or not example_id or example_id in self.ids:
            raise ValueError("Unique nonempty example IDs required")
        if (
            gradient.ndim != 2
            or not gradient.numel()
            or gradient.numel() > self.max_scalars
            or not torch.isfinite(gradient).all()
            or (self.mean is not None and gradient.shape != self.mean.shape)
        ):
            raise ValueError("Finite compact gradients with consistent shape required")
        value = gradient.detach().to(device="cpu", dtype=torch.float64)
        squared = value.square().sum(1)
        count = len(self.ids) + 1
        if self.mean is None:
            self.mean = value.clone()
            self.mean_squared_norm = squared
        else:
            self.mean += (value - self.mean) / count
            self.mean_squared_norm += (squared - self.mean_squared_norm) / count
        self.ids.add(example_id)

    def report(self, delta=None):
        """Report coherence, plus signed first-order change for a supplied update.

        Coherence = ||mean g||² / mean ||g||², not mean pairwise cosine.
        Zero-gradient rows have undefined coherence and are represented as None.
        A negative dot product predicts loss improvement. This accepts the actual
        optimizer proposal; it does not replace Adam by a raw-gradient step.
        """
        if self.mean is None:
            raise ValueError("Cannot report an empty bank")
        squared_mean = self.mean.square().sum(1)
        result = {
            "examples": len(self.ids),
            "mean_squared_norm": self.mean_squared_norm.tolist(),
            "squared_mean_norm": squared_mean.tolist(),
            "coherence": [
                float(a / b) if b > 0 else None
                for a, b in zip(squared_mean, self.mean_squared_norm, strict=True)
            ],
        }
        if delta is not None:
            if delta.shape != self.mean.shape or not torch.isfinite(delta).all():
                raise ValueError("Finite update with matching compact coordinates required")
            changes = (self.mean * delta.detach().to(device="cpu", dtype=torch.float64)).sum(1)
            result["predicted_change_by_neuron"] = changes.tolist()
            result["predicted_change"] = float(changes.sum())
        return result
