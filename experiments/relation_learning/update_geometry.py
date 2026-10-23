"""Offline, retrospective geometry of saved sparse updates; no quality prediction.

Treat each selected neuron's update as a row, separately for every projection.
Compare the best identical-row approximation with the best rank-one approximation.
The latter permits different signed coefficients; it is not NeST's shared update.
Neither approximation is deployed or evaluated by this audit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def geometry(rows):
    rows = rows.detach().cpu().double()
    if rows.ndim != 2 or min(rows.shape) < 1 or not torch.isfinite(rows).all():
        raise ValueError("Expected nonempty finite neuron-by-coordinate matrix")
    gram = rows @ rows.T
    energy = float(gram.trace())
    shared = float(gram.sum()) / len(rows)
    spectrum = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    rank_one = float(spectrum[0])
    return {
        "neurons": len(rows),
        "coordinates": rows.shape[1],
        "energy": energy,
        "identical_row_energy": shared,
        "rank_one_energy": rank_one,
        "identical_row_fraction": shared / energy if energy else None,
        "rank_one_fraction": rank_one / energy if energy else None,
        "spectrum": spectrum.tolist(),
    }


def audit_adapter(directory):
    config_path, tensor_path = directory / "patch.json", directory / "patch.safetensors"
    config = json.loads(config_path.read_text())
    if config["method"] != "paired_mlp" or config["format_version"] != 1:
        raise ValueError("This audit requires a full paired-MLP artifact")
    tensors = load_file(str(tensor_path))
    records = []
    seen = set()
    for selection in config["selections"]:
        indices = selection["indices"]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Unique nonempty support required")
        for name in (*selection["incoming"], selection["outgoing"]):
            if name in seen:
                raise ValueError("Duplicate projection")
            seen.add(name)
            rows = tensors[name].T if name == selection["outgoing"] else tensors[name]
            if rows.ndim != 2 or len(rows) != len(indices):
                raise ValueError("Manifest and tensor support disagree")
            records.append({"projection": name, **geometry(rows)})
    if seen != set(tensors):
        raise ValueError("Unexpected tensor keys")
    aggregate = {}
    for kind in ("all", "gate_proj", "up_proj", "down_proj"):
        subset = [r for r in records if kind == "all" or r["projection"].endswith(kind)]
        total = sum(r["energy"] for r in subset)
        aggregate[kind] = {
            "energy": total,
            **{
                metric + "_fraction": sum(r[metric + "_energy"] for r in subset) / total
                if total
                else None
                for metric in ("identical_row", "rank_one")
            },
        }
    return {
        "condition": config["metadata"]["condition"],
        "base_fingerprint": config["base_fingerprint"],
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (config_path, tensor_path)
        },
        "aggregate": aggregate,
        "projections": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(args.results.glob("case-0-paired_*-adapter"))
    if not directories:
        raise ValueError("No paired adapters found")
    result = {
        "interpretation": "Post-hoc geometry; not behavioral compression or training proof",
        "adapters": [audit_adapter(p) for p in directories],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
