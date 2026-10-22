"""Freeze a finite-update calibration before fitting a risk-aware recipe."""

import argparse
import json
from pathlib import Path

from experiments.paired_endpoint.prepare import sha256


def prepare(parent, integrity):
    proof = json.loads(integrity.read_text())
    assert proof["complete"] and proof["fits"] == 24
    identity = json.loads((parent / "identity.json").read_text())
    original = identity["protocol"]
    data = json.loads((parent / "inputs.json").read_text())
    assert len(data["cases"]) == 2 and {c["case"] for c in data["cases"]} == {0, 1}
    names = {"inputs.json", "identity.json", "completion.json", "tokenization.json"}
    for case in data["cases"]:
        names.update((f"case-{case['case']}-scores.safetensors", f"case-{case['case']}-base.json"))
    hashes = {name: sha256(parent / name) for name in sorted(names)}
    assert all(proof["file_sha256"][name] == value for name, value in hashes.items())
    return {
        "id": "functional-direction-calibration-v1-20261022",
        "parent_kernel": "shlydv/neuronpatch-paired-full-response-v1-20261022",
        "parent_files_sha256": hashes,
        "model": original["model"],
        "revision": original["revision"],
        "runtime": original["runtime"],
        "base_fingerprint": identity["base_fingerprint"],
        "architecture": original["architecture"],
        "inputs_sha256": original["inputs_sha256"],
        "seed": 43,
        "random_support_seed": 11,
        "rank_for_budget": 1,
        "shortlist_per_layer": 64,
        "fisher_probes_per_anchor": 2,
        "direction": "adam_first",
        "direction_epsilon": 1e-8,
        "damping_ratio": 1.0,
        "loss_scale": 128.0,
        "scales": [0.01, 0.03],
        "generation_scale": 0.03,
        "generation_banks": ["target_test", "scope_test", "utility_test"],
        "heldout_teacher_banks": ["scope_test", "utility_test"],
        "conditions": ["gradient_norm", "adam_gain", "gain_risk", "random"],
        "max_sequence_tokens": original["max_sequence_tokens"],
        "max_new_tokens": original["max_new_tokens"],
        "runtime_limit_seconds": 3600,
        "optimizer_trajectories": 0,
        "expected_one_step_edits": 2 * 4 * len(data["cases"]),
        "expected_scored_observations": 5
        * sum(
            len(c[b]) for c in data["cases"] for b in ("target_test", "scope_test", "utility_test")
        ),
        "expected_base_generation_controls": 8
        * sum(min(2, len(c["target_test"])) for c in data["cases"]),
        "fresh_confirmation_used": False,
        "direction_definition": (
            "Mean complete-response correction gradient over three original training phrases. "
            "Fresh unclipped Adam proposal -g/(abs(g)+eps),then unit-normalize each paired neuron "
            "across its incoming rows and outgoing column. This is a fixed direction family, "
            "not a simulated multi-step Adam fit. Apply selected directions at both frozen scales."
        ),
        "selection": (
            "Same per-layer neuron counts as the matched rank-one budget. Algorithmic candidates "
            "are the top64 original correction-gradient neurons per layer. Seed11 random supports "
            "are also captured but cannot affect risk-score damping or algorithmic selection. "
            "Compare norm-only,gain-only,gain/risk and random supports using identical proposals."
        ),
        "risk": (
            "Two independent locally seeded Rademacher softmax-covariance VJPs per original "
            "preservation response. Normalize by sqrt(response tokens),then equally across probes. "
            "Center at the frozen base; no gradient of zero reference KL is used. "
            "Rank individual gains squared over estimated curvature plus mean-curvature damping. "
            "Joint selected-support predictions sum probe dots before squaring."
        ),
        "contexts": (
            "Teacher contexts use the original canonical references,including EOS. These are fixed "
            "conditional-token distributions,not a full sequence-distribution Fisher estimate. "
            "Only the six preservation-training contexts influence scoring. Exposed scope/utility "
            "questions are held out of selection;their known outputs do not choose "
            "scale or support."
        ),
        "success": (
            "At scale0.03,in both development cases,gain_risk must reduce heldout mean anchored KL "
            "by at least20% versus adam_gain while retaining at least half its positive actual "
            "correction-NLL improvement,and itself improve correction NLL by at least0.001. "
            "It must also improve correction_gain_squared/heldout_KL by at least25%,with "
            "baseline heldout_KL>=1e-8 required for an informative comparison. "
            "At scale0.01,report joint predicted/actual calibration KL for every condition; "
            "a ratio outside[0.25,4] where actual KL>=1e-8 fails local numerical calibration. "
            "Require at least one informative small-scale condition in each case. "
            "Even success only licenses a trained-method pilot with LoRA,not a superiority claim."
        ),
        "limits": (
            "Two exposed facts,one small calibration bank,two sketches per anchor,shortlisted "
            "neurons and fixed initial directions. Sparse-group scores ignore between-group "
            "interactions during ranking;the joint estimate exposes those interactions. "
            "No multi-step optimizer is run;data-informed one-step edits are tested and restored. "
            "No robust fitted recipe is claimed,and no fresh cases are used. "
            "Record all conditions,quantization differences,unrelated answers and discovery cost."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = prepare(args.parent, args.integrity)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in spec.items() if k != "parent_files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
