"""Freeze a seven-fit stopping study using only exposed parent inputs/supports."""

import argparse
import hashlib
import json
from pathlib import Path

NORMAL_CONDITIONS = ("paired_mean-43-0", "paired_random-11-0", "lora-43-0")
DIAGNOSTIC = (0, "paired_mean-43-1")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(original, original_proof, replay, replay_proof):
    first = json.loads(original_proof.read_text())
    second = json.loads(replay_proof.read_text())
    assert first["complete"] and first["fits"] == 24
    assert second["complete"] and second["endpoints"] == 8
    replay_completion = json.loads((replay / "completion.json").read_text())
    assert second["decoded"] == replay_completion["observations"]
    identity = json.loads((original / "identity.json").read_text())
    parent_spec = identity["protocol"]
    data = json.loads((original / "inputs.json").read_text())
    fits = json.loads((original / "fits.json").read_text())
    original_names = {
        "identity.json",
        "inputs.json",
        "fits.json",
        "completion.json",
        "tokenization.json",
    }
    replay_names = {"identity.json", "protocol.json", "completion.json"}
    requests = []
    for case in data["cases"]:
        c = case["case"]
        original_names.update((f"case-{c}-base.json", f"case-{c}-scores.safetensors"))
        for condition in parent_spec["conditions"]:
            name = condition["id"]
            if name not in NORMAL_CONDITIONS and (c, name) != DIAGNOSTIC:
                continue
            fit = next(f for f in fits if f["case"] == c and f["condition"]["id"] == name)
            requests.append(
                {
                    "case": c,
                    "condition": condition,
                    "role": "comparison"
                    if name in NORMAL_CONDITIONS
                    else "known_failure_diagnostic",
                    "supports": fit["supports"],
                    "trainable_scalars": fit["trainable_scalars"],
                }
            )
            prefix = f"case-{c}-{name}"
            original_names.update((f"{prefix}.json", f"{prefix}-teacher.safetensors"))
            names = (
                ("adapter_config.json", "adapter_model.safetensors")
                if condition["family"] == "lora"
                else ("patch.json", "patch.safetensors")
            )
            original_names.update(f"{prefix}-adapter/{n}" for n in names)
            replay_names.add(f"{prefix}-live-teacher.safetensors")
    manifests = {}
    for group, root, names, proof in (
        ("original", original, original_names, first),
        ("replay", replay, replay_names, second),
    ):
        hashes = {n: sha256(root / n) for n in sorted(names)}
        assert all(proof["file_sha256"][n] == h for n, h in hashes.items())
        manifests[group] = hashes
    return {
        "id": "paired-checkpoint-study-v1-20261022",
        "kernel_sources": [
            "shlydv/neuronpatch-paired-full-response-v1-20261022",
            "shlydv/neuronpatch-paired-endpoint-v1-20261022",
        ],
        "parent_manifests": manifests,
        "model": parent_spec["model"],
        "revision": parent_spec["revision"],
        "runtime": parent_spec["runtime"],
        "base_fingerprint": identity["base_fingerprint"],
        "inputs_sha256": parent_spec["inputs_sha256"],
        "architecture": parent_spec["architecture"],
        "rank": parent_spec["rank"],
        "seed": parent_spec["seed"],
        "steps": parent_spec["steps"],
        "preservation_weight": parent_spec["preservation_weight"],
        "loss_scale": parent_spec["loss_scale"],
        "gradient_clip": parent_spec["gradient_clip"],
        "max_sequence_tokens": parent_spec["max_sequence_tokens"],
        "max_new_tokens": parent_spec["max_new_tokens"],
        "checkpoint_steps": [6, 12, 18, 24, 30, 36, 42, 48, 50],
        "gate_probability": 0.9,
        "gate": (
            "First observed post-update live checkpoint with every correction and auxiliary/"
            "preservation response token,including EOS,at probability>=0.9. "
            "No validation or test output chooses it."
        ),
        "requests": requests,
        "new_fits": len(requests),
        "fresh_confirmation_used": False,
        "generation_banks": ["target_test", "scope_test", "utility_test"],
        "runtime_limit_seconds": 6000,
        "predictions": [
            "Gradient-selected low-rate fits reach the observed confidence gate in fewer "
            "updates than random seed11 and LoRA;this need not mean lower total cost.",
            "Earlier adequate checkpoints reduce some fixed-step collateral damage;zero "
            "unrelated errors or unchanged transfer are not assumed.",
            "Matched grad-enabled and no-grad evaluations agree on target argmax at the "
            "same state;the known high-rate last update itself destabilizes its correction.",
            "Discovery is inexpensive relative to the old total sparse lifecycle,but total "
            "sparse cost may remain above LoRA because attachment/export/checking also count.",
        ],
        "fixed": (
            "Same original inputs,supports,FP16 base/FP32 trainables,Adam rates,cycling,"
            "loss scale,clipping,and50 updates. Continue after the first adequate checkpoint "
            "solely to recover the identical50-step reference. No norm/KL objective is added."
        ),
        "controls": (
            "Require all final trainable tensors,all50 historical training rows,live and merged "
            "final teacher logits,and final merged generations to exactly reproduce their parents. "
            "Otherwise report a failed trajectory-control check,not a stopping advantage."
        ),
        "execution_audit": (
            "At selected and final checkpoints compare grad-enabled and no-grad teacher outputs. "
            "For the known high-rate failure additionally audit the same final sampled prompt "
            "before and after the last update,and save parameters,gradient and Adam moments."
        ),
        "cost": (
            "Rerun discovery once per case and require frozen selected supports. Report discovery,"
            "training,monitoring,serialization and diagnostic costs separately. Counterfactual "
            "time-to-stop excludes the continued control suffix. No production throughput claim "
            "follows from a single observed trajectory."
        ),
        "useful_result": (
            "For the normal low-rate conditions,the selected-neuron early endpoint should retain"
            ">=7/8 complete transfer answers and have no new unambiguous preservation error on "
            "qualitative review. Its correction/preservation tradeoff and discovery-inclusive "
            "measured prefix cost must improve on the same stopping rule for random seed11 and "
            "official LoRA before advancing to fresh confirmation. Report all fixed50 and early "
            "outputs. The high-rate request is a diagnostic,not an extra tuned candidate."
        ),
        "limits": (
            "Two exposed development facts;single deliberately strong clean random seed. "
            "No paper superiority,novel loss or robust recipe. Existing exact-score labels remain "
            "primary;semantic aliases and ambiguous questions are reviewed separately. Auxiliary "
            "targets are unchanged even where the base initially misses them."
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("original", "original-proof", "replay", "replay-proof", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    spec = prepare(args.original, args.original_proof, args.replay, args.replay_proof)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(
        json.dumps(
            {
                "new_fits": spec["new_fits"],
                "requests": [
                    (r["case"], r["condition"]["id"], r["role"]) for r in spec["requests"]
                ],
                "checkpoint_steps": spec["checkpoint_steps"],
                "gate_probability": spec["gate_probability"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
