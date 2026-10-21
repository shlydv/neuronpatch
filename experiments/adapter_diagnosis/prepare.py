"""Freeze selected parent adapters and a complete no-refitting intervention grid."""

import argparse
import hashlib
import json
from pathlib import Path

from .groups import conditions
from .run import verify_parent


def prepare(parent, integrity):
    def read(name):
        return json.loads((parent / name).read_text())

    verified = json.loads(integrity.read_text())
    identity, completion, inputs, fits = [
        read(n) for n in ("identity.json", "completion.json", "inputs.json", "fits.json")
    ]
    assert verified["sequences_checked"] == verified["decoded"] == 530
    assert verified["calibration_records_recomputed"] == 8
    selected = completion["selected"]["lora"]
    assert selected == verified["selected"]["lora"] == "lora-low"
    assert completion["status"] == "complete" and completion["weights_restored"]
    source = identity["protocol"]
    records = [r for r in fits if r["condition"] == selected]
    assert len(records) == len(inputs["cases"]) == 2
    names = records[0]["metadata"]["targets"]
    assert all(r["metadata"]["targets"] == names for r in records)
    variants = conditions(names, source["model_dimensions"]["layers"])
    files = ["identity.json", "completion.json", "inputs.json", "fits.json"]
    for case in inputs["cases"]:
        c = case["case"]
        files += [
            f"case-{c}-base.json",
            f"case-{c}-{selected}.json",
            f"case-{c}-calibration-base.safetensors",
            f"case-{c}-{selected}-calibration.safetensors",
            f"case-{c}-{selected}-edit/adapter_config.json",
            f"case-{c}-{selected}-edit/adapter_model.safetensors",
        ]
    hashes = {name: hashlib.sha256((parent / name).read_bytes()).hexdigest() for name in files}
    assert all(
        verified["file_sha256"]["results/" + name] == digest for name, digest in hashes.items()
    )
    verify_parent(parent, hashes)
    observations = sum(
        len(c["edit_probes"])
        + len(inputs["utility"])
        + sum(len(c[k]) for k in ("calibration", "evaluation", "exposed_scope"))
        for c in inputs["cases"]
    )
    return {
        "id": "adapter-diagnosis-v1-20261021",
        "scope": "Two exposed development edits, frozen selected adapters; no fresh confirmation",
        **{k: source[k] for k in ("model", "revision", "base_fingerprint", "runtime")},
        "parent_kernel": "shlydv/neuronpatch-adapter-comparison-v1-20261021",
        "parent_accepted_version": 2,
        "parent_condition": selected,
        "parent_sha256": hashes,
        "parent_integrity_sha256": hashlib.sha256(integrity.read_bytes()).hexdigest(),
        "conditions": variants,
        "new_fits": 0,
        "new_training_steps": 0,
        "expected_interventions": len(variants) * len(inputs["cases"]),
        "expected_observations": (len(variants) + 1) * observations,
        "expected_control_observations": 2 * observations,
        "expected_ablation_observations": (len(variants) - 1) * observations,
        "control_policy": (
            "All base/full generations and calibration tensors exactly replay before ablation"
        ),
        "inference": "Same prompts, BOS, greedy decoding and40-token cap as parent",
        "runtime_limit_seconds": 3600,
        "criteria": {
            **source["selection"],
            "minimum_untrained_prefixes": 5,
            "maximum_evaluation_leaks": 1,
        },
        "interpretation": {
            "sufficient_subset": (
                "Report both edits' target NLL<=-log(.9), every calibration KL<=.1, "
                "2/2 original prefixes,>=5/6 untrained prefixes,<=1/8 evaluation target leaks "
                "and no lost baseline-reference evaluation/utility prefixes. A count pass "
                "is conditional evidence on this frozen adapter, not a usable recipe."
            ),
            "necessary_group": (
                "Report correction and preservation changes after group removal versus full. "
                "A failed removal does not establish a unique mechanism or that the group "
                "stores a concept; jointly fitted updates can compensate for each other."
            ),
            "decision": (
                "Outgoing-only sufficiency would weaken the missing-input-transform hypothesis; "
                "incoming/attention/final-layer dependence would prioritize a targeted "
                "parameterization diagnostic. No new selector, refit or rate search in this run."
            ),
            "quality": (
                "Retain all capped continuations; review later errors and target mentions "
                "as well as prefixes"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = prepare(args.parent, args.integrity)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: protocol[k]
                for k in (
                    "id",
                    "new_fits",
                    "expected_interventions",
                    "expected_observations",
                    "expected_control_observations",
                    "expected_ablation_observations",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
