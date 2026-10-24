"""Inspect saved grouped patches without loading a model or evaluating answers.

Effective update norms count a shared delta once per affected row. These are
pre-merge parameter-space measurements, not activation effects or causal scores.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from neuronpatch.grouped import validate_groups


def inspect(directory):
    directory = Path(directory)
    manifest_path = directory / "patch.json"
    tensor_path = directory / "patch.safetensors"
    manifest = json.loads(manifest_path.read_text())
    if manifest["method"] != "linear_grouped_output_rows":
        raise ValueError("Expected a grouped row adapter")
    tensors = load_file(tensor_path)
    if set(tensors) != set(manifest["targets"]) or set(tensors) != set(manifest["groups"]):
        raise ValueError("Adapter tensors, targets and groups must match")
    projections = {}
    layer_energy = {}
    for name, delta in sorted(tensors.items()):
        indices = manifest["targets"][name]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate selected rows")
        labels = validate_groups(manifest["groups"][name], len(indices))
        counts = torch.bincount(torch.tensor(labels))
        if delta.ndim != 2 or len(delta) != len(counts) or not torch.isfinite(delta).all():
            raise ValueError("Invalid group deltas")
        energy = float((delta.double().square().sum(1) * counts).sum())
        layer = int(name.split(".")[2])
        layer_energy[layer] = layer_energy.get(layer, 0.0) + energy
        projections[name] = {"group_sizes": counts.tolist(), "effective_squared_norm": energy}
    total = sum(layer_energy.values())
    minorities = sorted(min(row["group_sizes"]) for row in projections.values())
    return {
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest_path, tensor_path)
        },
        "projection_count": len(projections),
        "singleton_group_projections": minorities.count(1),
        "minority_at_most_four_projections": sum(n <= 4 for n in minorities),
        "minority_sizes_sorted": minorities,
        "effective_frobenius_norm": total**0.5,
        "layer_energy_fraction": {
            str(layer): energy / total if total else None
            for layer, energy in sorted(layer_energy.items())
        },
        "projections": projections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    choices = json.loads((args.results / "choices.json").read_text())
    report = {
        "scope": (
            "Saved grouped adapters only; pre-merge norms, no model inference or causal attribution"
        ),
        "adapters": {
            label: inspect(args.results / f"{label}-adapter")
            for label in choices
            if (args.results / f"{label}-adapter" / "patch.json").exists()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
