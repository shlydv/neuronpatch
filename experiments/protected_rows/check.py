"""Offline nonlinear/serialization check; no language-model quality claim."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from prototype import ProtectedRowEdit, protected_span
from torch import nn

from neuronpatch import NeuronPatch


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(12, 16)
        self.second = nn.Linear(16, 12)
        self.output = nn.Linear(12, 4)

    def forward(self, x):
        return self.output(self.second(self.first(x).tanh()).tanh())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    base = Network().double().eval()
    fresh = deepcopy(base)
    original = deepcopy(base.state_dict())
    anchors = torch.randn(3, 12, dtype=torch.float64)
    targets = {"first": [0, 2, 5, 8], "second": [1, 3, 6]}
    with torch.no_grad():
        spans = {
            "first": protected_span(anchors),
            "second": protected_span(base.first(anchors).tanh()),
        }
        anchor_reference = base(anchors).clone()
    teacher = deepcopy(base)
    with torch.no_grad():
        for name, indices in targets.items():
            weight = teacher.get_submodule(name).weight
            raw = torch.randn(len(indices), weight.shape[1], dtype=torch.float64) * 0.2
            basis = spans[name].basis
            weight[indices] += raw - (raw @ basis) @ basis.T
    corrections = torch.randn(24, 12, dtype=torch.float64)
    unseen = torch.randn(12, 12, dtype=torch.float64)
    with torch.no_grad():
        wanted = teacher(corrections)
        wanted_unseen = teacher(unseen)
        initial = float((base(corrections) - wanted).square().mean())
        initial_unseen = float((base(unseen) - wanted_unseen).square().mean())
    edit = ProtectedRowEdit(base, targets, spans).eval()
    optimizer = torch.optim.Adam(edit.trainable_parameters(), lr=0.03)
    max_anchor_change = 0
    for _ in range(100):
        loss = (edit(corrections) - wanted).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in edit.trainable_parameters())
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            max_anchor_change = max(
                max_anchor_change, float((edit(anchors) - anchor_reference).abs().max())
            )
    with torch.no_grad():
        final = float((edit(corrections) - wanted).square().mean())
        final_unseen = float((edit(unseen) - wanted_unseen).square().mean())
    assert final < initial * 0.1, (initial, final)
    assert max_anchor_change < 1e-12
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    try:
        edit(torch.randn(1, 13, dtype=torch.float64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected invalid-input failure")
    assert not edit._active
    assert not any(module._forward_hooks for module in base.modules())
    try:
        protected_span(torch.eye(12, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("A full protected span should have no editing capacity")
    inputs = torch.cat([anchors, corrections, unseen])
    with torch.no_grad():
        expected = edit(inputs)
    allocated, effective = edit.num_trainable_parameters, edit.effective_dimensions
    patch = edit.export()
    with TemporaryDirectory() as directory:
        destination = Path(directory) / "edit"
        patch.save(
            destination, metadata={"experiment": "protected_rows", "scope": "tiny FP64 check"}
        )
        loaded = NeuronPatch.load(fresh, destination)
        torch.testing.assert_close(loaded.model(inputs), expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(patch.merged_model()(inputs), expected, atol=1e-12, rtol=1e-12)
    result = {
        "status": "passed",
        "scope": (
            "tiny FP64 nonlinear fitting, anchor invariance and deployment; no LLM quality claim"
        ),
        "torch": torch.__version__,
        "initial_mse": initial,
        "final_mse": final,
        "initial_unseen_mse": initial_unseen,
        "final_unseen_mse": final_unseen,
        "max_calibration_anchor_change": max_anchor_change,
        "allocated_parameters": allocated,
        "effective_linear_dimension": effective,
        "protected_ranks": {name: span.rank for name, span in spans.items()},
        "base_weights": "unchanged",
        "reload_and_merge": "outputs match within 1e-12",
        "full_span_rejected": True,
        "exception_hooks_removed": True,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("prototype.py"))
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
