"""Inspect learned changes at fixed supports; no inference or causal norm control."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

from experiments.fine_transfer.delta_geometry import compare, read_columns


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.integrity.read_text())["file_sha256"]
    rows = []
    for case in (0, 1):
        base = args.results / f"case-{case}-w0-edit"
        first = load_file(str(base / "edit.safetensors"))["delta"].double()
        for weight in (1, 10):
            other = args.results / f"case-{case}-w{weight}-edit"
            for path in (base, other):
                for name in ("edit.json", "edit.safetensors"):
                    key = "results/" + str((path / name).relative_to(args.results))
                    assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected[key]
            second = load_file(str(other / "edit.safetensors"))["delta"].double()
            assert first.shape == second.shape
            rows.append(
                {
                    "case": case,
                    "weight": weight,
                    "occurrence_norm_unprotected": float(first.norm()),
                    "occurrence_norm_protected": float(second.norm()),
                    "occurrence_norm_ratio": float(second.norm() / first.norm()),
                    "linear_summed_columns": compare(read_columns(base), read_columns(other)),
                }
            )
    result = {
        "scope": "Post-hoc geometry; no extra model calls or fits",
        "comparisons": rows,
        "limitations": (
            "Duplicate occurrences summed linearly in float64, not actual sequential "
            "FP16 weight changes. Direction/norm differences do not prove why behavior "
            "changes. No dose-matched causal control has run."
        ),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "case": r["case"],
                    "weight": r["weight"],
                    "norm_ratio": r["occurrence_norm_ratio"],
                    "summed_update_cosine": r["linear_summed_columns"][
                        "whole_sparse_update_cosine"
                    ],
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
