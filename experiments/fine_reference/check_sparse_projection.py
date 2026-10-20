"""Test whether sparse masking preserves a projection constraint.

Independent linear-algebra diagnostic inspired by the audited NMKE operation
order. No research implementation is copied or executed. Random matrices are
not language-model activations, and this does not demonstrate a model advantage.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def ridge_update(keys, residual, ridge):
    gram = keys.T @ keys + ridge * torch.eye(keys.shape[1], dtype=keys.dtype)
    return keys @ torch.linalg.solve(gram, residual)


def fixture(seed, *, features=128, selected=48, preserved=12, edits=4, outputs=16):
    generator = torch.Generator().manual_seed(seed)
    protected = torch.linalg.qr(
        torch.randn(features, preserved, generator=generator, dtype=torch.float64)
    ).Q
    keys = torch.randn(features, edits, generator=generator, dtype=torch.float64)
    residual = torch.randn(edits, outputs, generator=generator, dtype=torch.float64)
    indices = torch.randperm(features, generator=generator)[:selected].sort().values
    ridge = 1e-4

    # Initial-edit AlphaEdit form with zero historical cache:
    # (P K K^T + ridge I)^(-1) P K R = P K (K^T P K + ridge I)^(-1) R.
    projected_keys = keys - protected @ (protected.T @ keys)
    full = projected_keys @ torch.linalg.solve(
        keys.T @ projected_keys + ridge * torch.eye(edits, dtype=keys.dtype), residual
    )
    post_mask = torch.zeros_like(full)
    post_mask[indices] = full[indices]

    # Solve in the intersection of the sparse support and the protected null space.
    restricted = protected[indices].T
    _, singular, vh = torch.linalg.svd(restricted, full_matrices=True)
    tolerance = max(restricted.shape) * torch.finfo(restricted.dtype).eps * singular.max()
    rank = int((singular > tolerance).sum())
    null_basis = vh[rank:].T.contiguous()
    edit_keys = null_basis.T @ keys[indices]
    joint = torch.zeros_like(full)
    joint[indices] = null_basis @ ridge_update(edit_keys, residual, ridge)

    # A support-only refit isolates the additional cost of preservation.
    support_only = torch.zeros_like(full)
    support_only[indices] = ridge_update(keys[indices], residual, ridge)

    # An output-row mask acts on the other axis and retains input protection.
    output_indices = torch.randperm(outputs, generator=generator)[: outputs // 2]
    output_masked = torch.zeros_like(full)
    output_masked[:, output_indices] = full[:, output_indices]
    outside_outputs = torch.ones(outputs, dtype=torch.bool)
    outside_outputs[output_indices] = False
    row_axis = {
        "protected_response_change": float(torch.linalg.vector_norm(protected.T @ output_masked)),
        "outside_output_support_norm": float(
            torch.linalg.vector_norm(output_masked[:, outside_outputs])
        ),
        "scope": "output-mask/input-projection compatibility only; different target output support",
    }

    rows = {}
    outside = torch.ones(features, dtype=torch.bool)
    outside[indices] = False
    for label, update in (
        ("projected_dense", full),
        ("projected_then_masked", post_mask),
        ("support_only_refit", support_only),
        ("joint_sparse_protected", joint),
    ):
        rows[label] = {
            "relative_edit_residual": float(
                torch.linalg.vector_norm(keys.T @ update - residual)
                / torch.linalg.vector_norm(residual)
            ),
            "protected_response_change": float(torch.linalg.vector_norm(protected.T @ update)),
            "outside_support_norm": float(torch.linalg.vector_norm(update[outside])),
            "update_norm": float(torch.linalg.vector_norm(update)),
        }
    return {
        "seed": seed,
        "features": features,
        "selected": selected,
        "protected_rank": preserved,
        "restricted_protected_rank": rank,
        "free_sparse_directions": null_basis.shape[1],
        "edit_directions": edits,
        "ridge": ridge,
        "conditions": rows,
        "output_row_axis_check": row_axis,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    rows = [fixture(seed) for seed in range(16)]
    no_capacity = fixture(100, selected=8, preserved=16)
    for row in rows:
        results = row["conditions"]
        assert results["projected_dense"]["protected_response_change"] < 1e-10
        assert results["projected_then_masked"]["protected_response_change"] > 1e-4
        assert results["joint_sparse_protected"]["protected_response_change"] < 1e-10
        assert results["joint_sparse_protected"]["outside_support_norm"] == 0
        assert results["joint_sparse_protected"]["relative_edit_residual"] < 1e-3
        assert results["projected_then_masked"]["relative_edit_residual"] > 0.1
        assert row["output_row_axis_check"]["protected_response_change"] < 1e-10
        assert row["output_row_axis_check"]["outside_output_support_norm"] == 0
    assert no_capacity["free_sparse_directions"] == 0
    assert no_capacity["conditions"]["joint_sparse_protected"]["relative_edit_residual"] == 1
    result = {
        "status": "passed",
        "scope": "FP64 linear-algebra fixtures; no language-model effectiveness claim",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "finding": (
            "Post-masking can break the protected-subspace constraint. Solving jointly "
            "can satisfy both constraints when enough sparse directions remain."
        ),
        "limits": (
            "A protected activation subspace is a proxy. Deep nonlinear propagation, "
            "multiple edited layers, finite precision and unseen inputs need separate checks. "
            "A sparse support may leave no protected editing capacity."
        ),
        "fixtures": rows,
        "infeasible_support": no_capacity,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "fixtures": len(rows),
                "scope": result["scope"],
                "mean_masked_protected_change": sum(
                    row["conditions"]["projected_then_masked"]["protected_response_change"]
                    for row in rows
                )
                / len(rows),
                "max_joint_protected_change": max(
                    row["conditions"]["joint_sparse_protected"]["protected_response_change"]
                    for row in rows
                ),
                "infeasible_support_detected": no_capacity["free_sparse_directions"] == 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
