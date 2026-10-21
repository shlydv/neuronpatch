"""Inspect frozen factor geometry without base weights or model inference.

Algebraic weight energy is not activation-weighted or causal importance. These
post-freeze measurements never change the fitted factors or intervention grid.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .run import verify_parent


def coordinate_energy(a, b, axis):
    """Squared row/column norms of B@A via small rank-space Gram matrices."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Compatible two-dimensional LoRA factors required")
    a, b = a.double().cpu(), b.double().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Finite factors required")
    if axis == "rows":
        energy = (b * (b @ (a @ a.T))).sum(1)
    elif axis == "columns":
        energy = (a * ((b.T @ b) @ a)).sum(0)
    else:
        raise ValueError("Choose rows or columns")
    if not torch.isfinite(energy).all() or (energy < 0).any():
        raise FloatingPointError("Invalid squared coordinate norms")
    return energy


def inspect(parent, protocol):
    verify_parent(parent, protocol["parent_sha256"])
    inputs = json.loads((parent / "inputs.json").read_text())
    rows = []
    for case in inputs["cases"]:
        c = case["case"]
        path = parent / f"case-{c}-{protocol['parent_condition']}-edit/adapter_model.safetensors"
        factors = load_file(str(path))
        energies, units = {}, {"incoming_rows": [], "outgoing_columns": []}
        for name in protocol["conditions"][0]["keep"]:
            a = factors[f"base_model.model.{name}.lora_A.weight"]
            b = factors[f"base_model.model.{name}.lora_B.weight"]
            column = coordinate_energy(a, b, "columns")
            energies[name] = float(column.sum())
            if name.endswith("mlp.fc_in"):
                units["incoming_rows"].append(coordinate_energy(a, b, "rows"))
            elif name.endswith("mlp.fc_out"):
                units["outgoing_columns"].append(column)
        total = sum(energies.values())
        if total <= 0:
            raise ValueError("A nonzero fitted adapter is required")
        coordinates = {}
        for name, values in units.items():
            energy = torch.cat(values).sort(descending=True).values
            cumulative = energy.cumsum(0) / energy.sum()
            coordinates[name] = {
                "coordinates": len(energy),
                "squared_norm": float(energy.sum()),
                "top_coordinate_energy_fraction": {
                    str(k): float(energy[:k].sum() / energy.sum())
                    for k in (5, 20, 32, 128, 512, 2048)
                },
                "coordinates_for_energy_fraction": {
                    str(q): int(torch.searchsorted(cumulative, q)) + 1 for q in (0.5, 0.9, 0.99)
                },
            }
        rows.append(
            {
                "case": c,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "full_squared_frobenius": total,
                "conditions": [
                    {
                        "condition": v["id"],
                        "projections": len(v["keep"]),
                        "fraction_of_full_squared_norm": sum(energies[n] for n in v["keep"])
                        / total,
                    }
                    for v in protocol["conditions"]
                ],
                "neuron_coordinates": coordinates,
            }
        )
    return {
        "status": "Post-freeze CPU weight geometry, not an inference result or selection change",
        "definition": (
            "Algebraic BA norms in float64 from saved FP32 factors; not rounded FP16 merge "
            "differences, activation effects or causal importance. Final-layer groups overlap "
            "projection families. Weight-energy pruning is not a behavioral compression test."
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.parent, json.loads(args.protocol.read_text()))
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Inspected {len(result['rows'])} frozen adapters without model inference")


if __name__ == "__main__":
    main()
