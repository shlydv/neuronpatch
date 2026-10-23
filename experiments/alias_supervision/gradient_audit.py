"""Describe recorded gradient contributions after independent result verification."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def audit(results, integrity):
    proof = json.loads(integrity.read_text())
    assert proof["complete"]
    for name, digest in proof["file_sha256"].items():
        assert hashlib.sha256((results / name).read_bytes()).hexdigest() == digest
    protocol = json.loads((results / "protocol.json").read_text())
    records = json.loads((results / "records.json").read_text())
    summaries = []
    for record in records:
        grouped = {"eligible": [], "unresolved_reference": []}
        maximum_norm_error = 0.0
        for history, trace in zip(record["history"], record["gradient_diagnostics"], strict=True):
            assert history["step"] == trace["step"]
            task = trace["task_gradient_norm"]
            auxiliary = trace["auxiliary_contribution_norm"]
            dot = trace["task_auxiliary_dot"]
            assert all(math.isfinite(v) for v in (task, auxiliary, dot))
            squared = task * task + auxiliary * auxiliary + 2 * dot
            assert squared >= -1e-10
            reconstructed = math.sqrt(max(0.0, squared))
            observed = history["gradient_norm"]
            error = abs(reconstructed - observed)
            assert error <= 2e-5 * max(reconstructed, observed) + 1e-7
            maximum_norm_error = max(maximum_norm_error, error)
            index = history["example_indices"][1]
            eligible = protocol["eligibility"][str(record["case"])][index]
            grouped["eligible" if eligible else "unresolved_reference"].append(
                {
                    "task": task,
                    "auxiliary": auxiliary,
                    "conflict": dot < 0,
                    "clipped": observed > protocol["gradient_clip"],
                }
            )
        summary = {}
        for name, rows in grouped.items():
            summary[name] = {
                "steps": len(rows),
                "median_task_norm": statistics.median(r["task"] for r in rows) if rows else None,
                "median_auxiliary_norm": (
                    statistics.median(r["auxiliary"] for r in rows) if rows else None
                ),
                "opposing_gradient_steps": sum(r["conflict"] for r in rows),
                "globally_clipped_steps": sum(r["clipped"] for r in rows),
            }
        summaries.append(
            {
                "case": record["case"],
                "method": record["method"],
                "arm": record["arm"],
                "maximum_reconstructed_norm_error": maximum_norm_error,
                "groups": summary,
            }
        )
    return {
        "post_hoc_descriptive": True,
        "rows": summaries,
        "limits": (
            "Auxiliary contribution is cumulative-minus-task gradient before clipping, including "
            "accumulation rounding. Different arms visit different parameter states. Norms and "
            "cosines do not identify EOS-specific mediation, predict Adam displacement, or "
            "establish a neuron-selection advantage. Primary outcome criteria are unchanged."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("results", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.results, args.integrity)
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"fits_audited": len(value["rows"])}))


if __name__ == "__main__":
    main()
