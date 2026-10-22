"""Freeze a changed-method development pilot after the failed quadratic-risk gate."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(parent, calibration, integrity):
    proof = json.loads(integrity.read_text())
    assert proof["complete"] and proof["one_step_edits"] == 16
    records_path = calibration / "records.json"
    assert (
        hashlib.sha256(records_path.read_bytes()).hexdigest()
        == proof["file_sha256"]["records.json"]
    )
    old = json.loads((calibration / "protocol.json").read_text())
    assert (
        hashlib.sha256((calibration / "protocol.json").read_bytes()).hexdigest()
        == proof["file_sha256"]["protocol.json"]
    )
    for name, digest in old["parent_files_sha256"].items():
        assert hashlib.sha256((parent / name).read_bytes()).hexdigest() == digest
    data = json.loads((parent / "inputs.json").read_text())
    records = json.loads(records_path.read_text())
    methods = ["gradient_norm", "gain_risk", "random", "lora"]
    supports = {
        str(c["case"]): {
            method: next(
                r["selections"]
                for r in records
                if r["case"] == c["case"] and r["condition"] == method and r["scale"] == 0.03
            )
            for method in methods
            if method != "lora"
        }
        for c in data["cases"]
    }
    banks = ["target_test", "scope_test", "utility_test"]
    return {
        "id": "measured-preservation-training-v1-20261022",
        "parent_kernel": old["parent_kernel"],
        "parent_files_sha256": old["parent_files_sha256"],
        "model": old["model"],
        "revision": old["revision"],
        "runtime": old["runtime"],
        "base_fingerprint": old["base_fingerprint"],
        "architecture": old["architecture"],
        "rank": 1,
        "seed": 43,
        "methods": methods,
        "supports": supports,
        "support_provenance": {
            "calibration_records_sha256": proof["file_sha256"]["records.json"],
            "provider_version": 2,
            "algorithm_frozen_before_previous_inference": True,
            "cached_development_supports": True,
            "rediscovery_in_this_run": False,
            "limits": (
                "Initial fixed supports only. Prior discovery cost must be counted "
                "separately;this run cannot establish fresh selection generalization or "
                "end-to-end discovery timing."
            ),
        },
        "steps": 60,
        "learning_rate": 0.0003,
        "preservation_weight": 1.0,
        "loss_scale": 128.0,
        "gradient_clip": 1.0,
        "confidence": 0.9,
        "backtrack_scales": [1.0, 0.5, 0.25, 0.125],
        "budget": {"mean_kl": 0.02, "worst_kl": 0.2, "minimum_improvement": 1e-7},
        "max_sequence_tokens": old["max_sequence_tokens"],
        "max_new_tokens": old["max_new_tokens"],
        "generation_banks": banks,
        "heldout_teacher_banks": ["scope_test", "utility_test"],
        "expected_fits": len(methods) * len(data["cases"]),
        "expected_observations": (1 + 3 * len(methods))
        * sum(len(c[b]) for c in data["cases"] for b in banks),
        "expected_base_generation_controls": len(methods)
        * sum(min(2, len(c["target_test"])) for c in data["cases"]),
        "runtime_limit_seconds": 3000,
        "provider_timeout_seconds": 5400,
        "fresh_confirmation_used": False,
        "objective": (
            "Equal-example full-response correction CE plus KL(base||current) on the "
            "original six canonical preservation contexts. Full-batch gradients each "
            "proposal;all representations use identical measured acceptance and stopping."
        ),
        "stopping": (
            "Stop at every correction token including EOS probability>=0.9,60 "
            "proposals,or failure of all four backtracks. No held-out output chooses a "
            "checkpoint or retries. No accepted update is a valid unsuccessful "
            "adaptation outcome."
        ),
        "preservation": (
            "Measure absolute mean and worst-token anchored KL from the frozen base "
            "after each provisional Adam update;require task-loss improvement. Fully "
            "restore rejected parameters,gradients,Adam state and RNG. Same numerical "
            "limits apply to every method. No held-out teacher bank or generated answer "
            "influences acceptance."
        ),
        "comparison": (
            "Official rank-one MLP LoRA and random seed11 are known strong controls from "
            "the prior completed study. Use the shared previously successful low "
            "rate0.0003;do not select a new rate after observing outputs. Report both "
            "live and merged complete answers after real save/reload."
        ),
        "success": (
            "A risk-aware candidate must achieve8/8 requested merged transfer answers across "
            "both exposed facts,with no new unambiguous preservation factual errors and no "
            "truncation,while satisfying both mean and worst preservation budgets after merging. "
            "Both matched random and official LoRA controls must also complete8/8 edits for an "
            "informative superiority comparison. At equal complete-answer success,the candidate "
            "must reduce held-out mean KL by>=20% versus each of those two controls in each case "
            "without worse factual preservation. If a control fails editing,the superiority "
            "comparison is inconclusive rather than an automatic win. Report the gradient-only "
            "ablation,all failed fits,and all phases of cost separately,including the cached "
            "support provenance;do not claim measured whole-pipeline speed from cached supports. "
            "Any positive result only licenses fresh confirmation,not a robust recipe."
        ),
        "limits": (
            "Changed method after an explicitly failed previous gate;that result remains "
            "failed. Two exposed development facts and canonical preservation contexts. "
            "No novel-optimizer claim or reproduction of paper training. Cached supports "
            "do not measure whole-pipeline speed. Full neuron selection,additional "
            "parameter/data budgets,another task family and fresh cases remain required."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "calibration", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.parent, args.calibration, args.integrity)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("supports", "parent_files_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
