"""Document-held-out recognition within the saved discovery distribution.

This does not measure new attack families, editing utility, or model answers.
Only saved training profiles are used; both variants of a document stay together.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .probe_selection import fit_probe


def run(root):
    root = Path(root)
    metadata = json.loads((root / "profile-metadata.json").read_text())
    ids = metadata["ids"]
    documents = [identifier.split(":")[0] for identifier in ids]
    unique = sorted(set(documents), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(unique) < 8 or len(set(ids)) != len(ids):
        raise ValueError("At least eight documents and unique case IDs required")
    assignments = {doc: i % 4 for i, doc in enumerate(unique)}
    folds = torch.tensor([assignments[doc] for doc in documents])
    labels = torch.tensor(metadata["labels"])
    profiles = load_file(root / "training-profiles.safetensors")
    records = []
    for name, values in sorted(profiles.items()):
        if len(values) != len(ids):
            raise ValueError("Profile and metadata row counts differ")
        for fold in range(4):
            train, test = folds != fold, folds == fold
            if set(labels[test].tolist()) != {0, 1}:
                raise ValueError("Every held-out fold must contain both classes")
            probe = fit_probe(values[train], labels[train])
            standardized = (values[test].double() - probe["mean"]) / probe["scale"]
            correct = int(
                ((standardized @ probe["weight"] + probe["bias"] >= 0) == labels[test].bool()).sum()
            )
            indices = torch.argsort(probe["weight"].abs(), descending=True, stable=True)[:32]
            sparse = fit_probe(values[train][:, indices], labels[train])
            sparse_x = (values[test][:, indices].double() - sparse["mean"]) / sparse["scale"]
            sparse_correct = int(
                ((sparse_x @ sparse["weight"] + sparse["bias"] >= 0) == labels[test].bool()).sum()
            )
            records.append(
                {
                    "projection": name,
                    "fold": fold,
                    "correct": correct,
                    "total": int(test.sum()),
                    "training_accuracy": probe["training_accuracy"],
                    "train_selected_32_correct": sparse_correct,
                }
            )
        print("CHECKED", name, flush=True)
    return {
        "scope": "Four document-held-out folds within original training discovery templates",
        "not_evidence_of": ["new-attack recognition", "useful update directions", "answer quality"],
        "files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("profile-metadata.json", "training-profiles.safetensors")
        },
        "documents": len(unique),
        "cases": len(ids),
        "folds_by_document": assignments,
        "correct": sum(row["correct"] for row in records),
        "train_selected_32_correct": sum(row["train_selected_32_correct"] for row in records),
        "total": sum(row["total"] for row in records),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.results), indent=2) + "\n")


if __name__ == "__main__":
    main()
