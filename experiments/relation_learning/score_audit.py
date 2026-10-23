"""Audit saved Fisher scores without loading the model or changing any support."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def summarize(values, supports):
    """Compare equal-cost neuron supports against their score-only optimum.

    Values are already squared per-example gradients averaged over examples;
    squaring again would compute a different quantity. No quality is predicted.
    """
    values = values.double()
    if values.ndim != 2 or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a finite nonnegative layer-by-neuron score matrix")
    total = float(values.sum())
    if total <= 0 or not supports:
        raise ValueError("Positive total score and recorded supports required")
    records = []
    budgets = set()
    for name, layers in supports.items():
        if len(layers) != values.shape[0]:
            raise ValueError("One index list per layer required")
        mass, budget = 0.0, 0
        for layer, indices in enumerate(layers):
            if len(set(indices)) != len(indices) or any(
                not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < values.shape[1]
                for i in indices
            ):
                raise ValueError("Unique in-range integer indices required")
            mass += float(values[layer, indices].sum())
            budget += len(indices)
        budgets.add(budget)
        records.append({"condition": name, "score_mass": mass, "fraction_total": mass / total})
    if len(budgets) != 1 or next(iter(budgets)) == 0:
        raise ValueError("Equal positive neuron budgets required")
    budget = budgets.pop()
    flat = values.flatten()
    top = torch.argsort(flat, descending=True, stable=True)[:budget]
    optimum = float(flat[top].sum())
    for record in records:
        record["fraction_global_score_optimum"] = record["score_mass"] / optimum
    return {
        "candidate_neurons": values.numel(),
        "selected_neurons": budget,
        "global_score_fraction_total": optimum / total,
        "global_score_layer_counts": torch.bincount(
            top // values.shape[1], minlength=values.shape[0]
        ).tolist(),
        "recorded_supports": records,
    }


def audit(root):
    paths = [root / name for name in ("fits.json", "case-0-scores.safetensors", "identity.json")]
    fits = json.loads(paths[0].read_text())
    scores = load_file(str(paths[1]))
    identity = json.loads(paths[2].read_text())
    counts = identity["layer_neuron_counts"]
    values = torch.stack([scores[f"{i}-paired_empirical_fisher"] for i in range(len(counts))])
    supports = {}
    for fit in fits:
        if fit["condition"]["family"] not in ("paired_random", "paired_fisher"):
            continue
        assert fit["case"] == 0 and len(fit["supports"]) == len(counts)
        layers = []
        for i, entry in enumerate(fit["supports"]):
            assert entry["outgoing"] == f"model.layers.{i}.mlp.down_proj"
            assert len(entry["indices"]) == counts[i]
            if fit["condition"]["family"] == "paired_fisher":
                assert (
                    entry["indices"]
                    == torch.argsort(values[i], descending=True, stable=True)[: counts[i]].tolist()
                )
            layers.append(entry["indices"])
        supports[fit["condition"]["id"]] = layers
    return {
        "scope": "Retrospective score arithmetic, not an intervention or fresh prediction",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "definition": "Sum of stored empirical-Fisher group scores; no additional squaring",
        **summarize(values, supports),
        "limitations": [
            "No signed gradients, cross-example agreement or Adam directions are stored here.",
            "Global score optimum is not a trained or behaviorally evaluated support.",
            "Score concentration does not establish semantic importance or safe adaptation.",
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
