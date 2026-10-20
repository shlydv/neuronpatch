"""Inspect verified FiNE update geometry without loading or changing a model.

SVD is an established compression tool, not a new algorithm. These are only two
exposed development edits and their random controls. Weight-space reconstruction
does not prove preserved generations, transfer or a cheaper training method.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def analyze(delta, targets):
    values = delta.double()
    total = values.square().sum()
    if total <= 0 or not torch.isfinite(values).all():
        raise ValueError("Nonzero finite update required")
    u, s, vh = torch.linalg.svd(values, full_matrices=False)
    cumulative = s.square().cumsum(0) / total
    entries = []
    for rank in sorted({1, 2, 4, 8, len(s)}):
        if rank > len(s):
            continue
        left = (u[:, :rank] * s[:rank]).to(delta.dtype)
        right = vh[:rank].to(delta.dtype)
        reconstructed = (left.float() @ right.float()).to(delta.dtype)
        entries.append(
            {
                "rank": rank,
                "retained_squared_frobenius_energy": float(cumulative[rank - 1]),
                "relative_error_after_bf16_factor_storage": float(
                    (reconstructed.double() - values).norm() / values.norm()
                ),
                "factor_scalars": left.numel() + right.numel(),
                "raw_scalars": delta.numel(),
                "factor_to_raw_scalar_ratio": (left.numel() + right.numel()) / delta.numel(),
            }
        )
    # Aggregate duplicate target occurrences only in exact linear arithmetic.
    # This is NOT the actual FP16 sequential weight merge, which also needs base columns.
    grouped = {}
    for row, target in zip(values, targets):
        key = tuple(target)
        grouped[key] = grouped.get(key, torch.zeros_like(row)) + row
    return {
        "occurrences": len(targets),
        "unique_columns": len(grouped),
        "modules": len({t[0] for t in targets}),
        "dtype": str(delta.dtype),
        "rank_for_95_percent_energy": int((cumulative < 0.95).sum()) + 1,
        "rank_for_99_percent_energy": int((cumulative < 0.99).sum()) + 1,
        "global_cross_layer_factorizations": entries,
        "singular_values": s.tolist(),
        "merge_caution": (
            "Stored occurrence updates only. Global shared basis spans layers. "
            "Actual merged-weight/generation equivalence has not been tested."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in sorted(args.source.glob("*edit/edit.safetensors")):
        manifest_path = path.with_name("edit.json")
        manifest = json.loads(manifest_path.read_text())
        delta = load_file(str(path))["delta"]
        assert delta.shape == (len(manifest["targets"]), 4096)
        result = analyze(delta, manifest["targets"])
        result.update(
            artifact=path.parent.name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        results.append(result)
    assert len(results) == 4
    report = {
        "status": "CPU geometry audit only; no model quality or novelty claim",
        "scope": "Two verified source-parity GPT-J-6B edits and matching random controls",
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in results:
        print(
            json.dumps(
                {
                    "artifact": r["artifact"],
                    "rank99": r["rank_for_99_percent_energy"],
                    "rank1": r["global_cross_layer_factorizations"][0],
                    "rank4": next(
                        (x for x in r["global_cross_layer_factorizations"] if x["rank"] == 4), None
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
