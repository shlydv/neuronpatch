"""Freeze eight saved endpoints and parent hashes; no fitting or new questions."""

import argparse
import hashlib
import json
from pathlib import Path

CONDITIONS = (
    "paired_mean-43-0",
    "paired_mean-43-1",
    "paired_random-11-0",
    "lora-43-0",
)
GENERATION_BANKS = ("target_test", "scope_test", "utility_test")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent, integrity):
    """Bind exactly the already verified files needed for this diagnostic."""
    proof = json.loads(integrity.read_text())
    assert proof["complete"] and proof["fits"] == 24
    identity = json.loads((parent / "identity.json").read_text())
    original = identity["protocol"]
    data = json.loads((parent / "inputs.json").read_text())
    fits = json.loads((parent / "fits.json").read_text())
    completion = json.loads((parent / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["base_restored"]
    assert completion["final_fingerprint"] == identity["base_fingerprint"]
    assert sha256(parent / "inputs.json") == original["inputs_sha256"]
    requested = []
    names = {"inputs.json", "identity.json", "fits.json", "completion.json", "tokenization.json"}
    for case in data["cases"]:
        c = case["case"]
        names.add(f"case-{c}-base.json")
        for condition in CONDITIONS:
            fit = [f for f in fits if f["case"] == c and f["condition"]["id"] == condition]
            assert len(fit) == 1 and fit[0]["reload_teacher_exact"] and fit[0]["base_restored"]
            requested.append({"case": c, "condition": fit[0]["condition"]})
            prefix = f"case-{c}-{condition}"
            names.update((f"{prefix}.json", f"{prefix}-teacher.safetensors"))
            adapter = parent / f"{prefix}-adapter"
            expected = (
                ("adapter_config.json", "adapter_model.safetensors")
                if fit[0]["condition"]["family"] == "lora"
                else ("patch.json", "patch.safetensors")
            )
            names.update(str((adapter / n).relative_to(parent)) for n in expected)
    hashes = {name: sha256(parent / name) for name in sorted(names)}
    assert all(proof["file_sha256"][n] == h for n, h in hashes.items())
    generations = sum(
        2 * len(CONDITIONS) * sum(len(c[b]) for b in GENERATION_BANKS) for c in data["cases"]
    )
    return {
        "id": "paired-endpoint-replay-v1-20261022",
        "parent_kernel": "shlydv/neuronpatch-paired-full-response-v1-20261022",
        "model": original["model"],
        "revision": original["revision"],
        "runtime": original["runtime"],
        "base_fingerprint": identity["base_fingerprint"],
        "inputs_sha256": original["inputs_sha256"],
        "parent_files_sha256": hashes,
        "endpoints": requested,
        "generation_banks": list(GENERATION_BANKS),
        "expected_generations": generations,
        "new_fits": 0,
        "fresh_confirmation_used": False,
        "runtime_limit_seconds": 2700,
        "max_new_tokens": original["max_new_tokens"],
        "max_sequence_tokens": original["max_sequence_tokens"],
        "hypotheses": [
            "A failed final merged answer need not imply a failed live training endpoint.",
            "The selected low-rate paired patch's leakage may exist before merging.",
            (
                "Random and official LoRA controls distinguish paired-specific "
                "from general replay issues."
            ),
        ],
        "decisions": (
            "Replay identical saved FP32 deltas on the same pinned FP16 base. Compare all complete "
            "training, preservation and validation teacher tokens before and after merging. "
            "Generate every existing target_test/scope_test/utility_test question in both states. "
            "Require merged teacher logits and generations to exactly reproduce parent artifacts. "
            "Any mismatch prevents attributing a live/merged difference to the original endpoint. "
            "No support, rate, prompt, target, score or parameter changes. "
            "No new selector is tested."
        ),
        "limits": (
            "Eight post-hoc development endpoints deliberately include a known failure. "
            "This is not unbiased method selection, fresh confirmation "
            "or general merge equivalence. "
            "This test cannot distinguish all finite-precision accumulation and casting causes."
        ),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--integrity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = prepare(args.parent, args.integrity)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in spec.items() if k != "parent_files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
