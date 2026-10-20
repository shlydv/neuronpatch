"""Inspect saved update directions without training or running a model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .scope_analysis import CONDITIONS


def read_columns(path):
    manifest = json.loads((path / "edit.json").read_text())
    delta = load_file(str(path / "edit.safetensors"))["delta"].double()
    assert len(manifest["targets"]) == len(delta) and torch.isfinite(delta).all()
    grouped = {}
    for key, vector in zip(manifest["targets"], delta, strict=True):
        key = tuple(key)
        grouped[key] = grouped.get(key, torch.zeros_like(vector)) + vector
    return grouped


def compare(a, b):
    """Compare common columns in residual coordinates; never semantic similarity."""
    energy_a = sum(float(v.square().sum()) for v in a.values())
    energy_b = sum(float(v.square().sum()) for v in b.values())
    rows = []
    overlap = sorted(a.keys() & b.keys())
    for module, feature in overlap:
        x, y = a[module, feature], b[module, feature]
        nx, ny = float(x.norm()), float(y.norm())
        rows.append(
            {
                "module": module,
                "feature": feature,
                "norm_a": nx,
                "norm_b": ny,
                "cosine": float(x @ y) / (nx * ny) if nx and ny else None,
            }
        )
    dot = sum(float(a[key] @ b[key]) for key in overlap)
    return {
        "unique_columns_a": len(a),
        "unique_columns_b": len(b),
        "shared_columns": rows,
        "shared_squared_norm_fraction_a": (
            sum(float(a[key].square().sum()) for key in overlap) / energy_a if energy_a else None
        ),
        "shared_squared_norm_fraction_b": (
            sum(float(b[key].square().sum()) for key in overlap) / energy_b if energy_b else None
        ),
        "whole_sparse_update_cosine": dot / (energy_a * energy_b) ** 0.5
        if energy_a and energy_b
        else None,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads((Path(__file__).parent / "scope_execution_v1.json").read_text())[
        "data_sha256"
    ]
    patches, hashes = {}, {}
    for case in (0, 1):
        for name in CONDITIONS:
            path = (
                args.references / f"case-{case}"
                if name == "fine-original"
                else args.pilot_results / f"case-{case}-{name}-edit"
            )
            for filename in ("edit.json", "edit.safetensors"):
                key = f"edits/case-{case}-{name}/{filename}"
                hashes[key] = hashlib.sha256((path / filename).read_bytes()).hexdigest()
                assert hashes[key] == expected[key], key
            patches[case, name] = read_columns(path)
    result = {
        "scope": "Post-hoc geometry of frozen two-case patches; no model inference or new fits",
        "limitations": (
            "Duplicate occurrences are summed linearly in float64, not sequentially merged "
            "into FP16 base weights. Cosines describe stored parameter updates in matching "
            "residual coordinates, not logits, semantic similarity or causal effects. Across "
            "different layers they cannot establish a common response direction. Two facts "
            "and overlapping columns are not independent samples."
        ),
        "different_fact_same_selector": {
            name: compare(patches[0, name], patches[1, name]) for name in CONDITIONS
        },
        "same_fact_different_selector": {
            str(case): compare(
                patches[case, "gradient_single"], patches[case, "gradient_consensus"]
            )
            for case in (0, 1)
        },
        "artifact_sha256": hashes,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("artifact_sha256", "limitations")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
