"""Exact parameter-space projection of a saved LoRA update onto grouped rows.

Uses small low-rank factors, never base weights or model inference. Frobenius
energy is not behavioral importance: a different update can implement the same
function, and a small parameter error can have a large activation effect.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from neuronpatch.grouped import validate_groups


def energies(a, b, indices, groups):
    """Squared norms of nested orthogonal projections of B @ A.

    With fixed disjoint row groups, the least-squares shared update is the mean
    target row in each group. A @ A.T evaluates its norm without expanding B @ A.
    The top-row quantity permits independent rows and is an oracle upper bound
    for any shared grouping with the same row count, not a deployable selector.
    """
    a, b = a.double(), b.double()
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Incompatible low-rank factors")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Finite factors required")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Nonempty unique row indices required")
    if any(type(i) is not int or not 0 <= i < len(b) for i in indices):
        raise ValueError("Row index out of range")
    labels = torch.tensor(validate_groups(groups, len(indices)))
    gram = a @ a.T
    row_energy = ((b @ gram) * b).sum(1)
    selected = b[indices]
    shared = 0.0
    for label in labels.unique():
        members = selected[labels == label]
        mean = members.mean(0)
        shared += len(members) * float(mean @ gram @ mean)
    return {
        "full": float(row_energy.sum()),
        "selected_independent": float(row_energy[indices].sum()),
        "selected_shared": shared,
        "oracle_top_rows_independent": float(row_energy.topk(len(indices)).values.sum()),
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(lora, patches):
    config_path = lora / "adapter_config.json"
    weights_path = lora / "adapter_model.safetensors"
    config = json.loads(config_path.read_text())
    if (
        config["peft_type"] != "LORA"
        or config.get("fan_in_fan_out")
        or config.get("use_dora")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
        or config.get("bias", "none") != "none"
        or config.get("lora_bias")
    ):
        raise ValueError("Only ordinary uniform-rank bias-free LoRA is supported")
    rank = config["r"]
    scale = config["lora_alpha"] / (rank**0.5 if config.get("use_rslora") else rank)
    tensors = load_file(weights_path)
    factors = {}
    used = set()
    for key in sorted(tensors):
        if not key.endswith(".lora_A.weight"):
            continue
        partner = key.replace(".lora_A.weight", ".lora_B.weight")
        name = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        factors[name] = (tensors[key].double() * scale, tensors[partner].double())
        if factors[name][0].shape[0] != rank:
            raise ValueError("Configured rank does not match saved factors")
        used.update((key, partner))
    if not factors or used != set(tensors):
        raise ValueError("Unaccounted adapter tensors")
    module_energy = {name: float(((b @ (a @ a.T)) * b).sum()) for name, (a, b) in factors.items()}
    total = sum(module_energy.values())
    if total <= 0:
        raise ValueError("Nonzero update required")
    reports = {}
    hashes = {"lora_config": sha(config_path), "lora_weights": sha(weights_path)}
    for label, directory in patches.items():
        path = directory / "patch.json"
        manifest = json.loads(path.read_text())
        if manifest["method"] != "linear_grouped_output_rows":
            raise ValueError("Expected grouped row support")
        if set(manifest["targets"]) != set(manifest["groups"]):
            raise ValueError("Support and group targets differ")
        per_module = {
            name: energies(*factors[name], indices, manifest["groups"][name])
            for name, indices in manifest["targets"].items()
        }
        sums = {
            key: sum(row[key] for row in per_module.values())
            for key in next(iter(per_module.values()))
        }
        reports[label] = {
            "squared_norms": sums,
            "fraction_of_entire_lora_energy": {k: v / total for k, v in sums.items()},
            "shared_fraction_of_selected_energy": (
                sums["selected_shared"] / sums["selected_independent"]
                if sums["selected_independent"]
                else None
            ),
            "minimum_relative_frobenius_error": (
                max(0.0, 1 - sums["selected_shared"] / total) ** 0.5
            ),
            "modules": per_module,
        }
        hashes[label + "_support_manifest"] = sha(path)
    return {
        "scope": "Exact weight-space least squares, not behavioral fidelity or a capacity proof",
        "files_sha256": hashes,
        "lora_squared_norm": total,
        "lora_module_squared_norms": module_energy,
        "supports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--patch", action="append", required=True, help="LABEL=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patches = {}
    for item in args.patch:
        label, directory = item.split("=", 1)
        if label in patches:
            raise ValueError("Duplicate support label")
        patches[label] = Path(directory)
    report = inspect(args.lora, patches)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
