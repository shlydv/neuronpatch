"""Audit existing discovery scores; no fresh model execution or support refitting."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def audit(root):
    fits_path = root / "fits.json"
    fits = json.loads(fits_path.read_text())
    assert len(fits) == 24
    source_paths = [fits_path]
    cases = []
    for case in (0, 1):
        score_path = root / f"case-{case}-scores.safetensors"
        discovery_path = root / f"case-{case}-discovery.json"
        source_paths.extend((score_path, discovery_path))
        scores = load_file(str(score_path))
        counts = json.loads(discovery_path.read_text())["layer_counts"]
        values = torch.stack(
            [scores[f"{i}-paired_gradient_norm"].double() for i in range(len(counts))]
        )
        assert torch.isfinite(values).all() and (values >= 0).all()
        flat = values.flatten()
        total = float(flat.square().sum())
        assert total > 0
        budget = sum(counts)
        top = torch.argsort(flat, descending=True, stable=True)[:budget]
        optimum = float(flat[top].square().sum())
        groups = []
        for fit in fits:
            condition = fit["condition"]
            if fit["case"] != case or not condition["id"].endswith("-0"):
                continue
            if condition["family"] not in ("paired_mean", "paired_random"):
                continue
            support = fit["supports"]
            assert len(support) == len(counts)
            selected = []
            for i, (entry, count) in enumerate(zip(support, counts, strict=True)):
                assert entry["outgoing"] == f"model.layers.{i}.mlp.down_proj"
                indices = entry["indices"]
                assert len(indices) == len(set(indices)) == count
                assert all(0 <= index < values.shape[1] for index in indices)
                if condition["family"] == "paired_mean":
                    expected = torch.argsort(values[i], descending=True, stable=True)[:count]
                    assert indices == expected.tolist()
                selected.append(values[i, indices].square().sum())
            mass = float(torch.stack(selected).sum())
            groups.append(
                {
                    "condition": condition["id"],
                    "fraction_squared_mean_gradient": mass / total,
                    "fraction_same_budget_global_score_optimum": mass / optimum,
                }
            )
        assert len(groups) == 4
        cases.append(
            {
                "case": case,
                "candidate_neurons": flat.numel(),
                "selected_neurons": budget,
                "global_score_optimum_fraction_squared_mean_gradient": optimum / total,
                "global_score_optimum_layer_counts": torch.bincount(
                    top // values.shape[1], minlength=len(counts)
                ).tolist(),
                "recorded_supports": groups,
            }
        )
    return {
        "scope": "Retrospective arithmetic on the two exposed cases' saved discovery scores.",
        "definition": (
            "Sum of squared Euclidean mean-correction-gradient norms over the selected "
            "gate/up rows and down columns, divided by that sum over all MLP neurons. "
            "Equal neuron costs make global top-k optimal for this recorded score."
        ),
        "cases": cases,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "limitations": [
            "This score is first-order Euclidean geometry, not actual Adam improvement.",
            "No preservation gradients, update trajectories or unseen outputs enter this score.",
            "A large score does not establish semantic localization or a behavioral advantage.",
            "Global supports are computed analytically but not trained or evaluated.",
            "All data were previously exposed; there is no fresh predictive validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["cases"], indent=2))


if __name__ == "__main__":
    main()
