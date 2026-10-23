"""Compare the completed v1 fit with v2 without displaying held-out answers."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def compare(before, after):
    label = "lora-43-0.0003"

    def fit(root):
        matches = [
            r
            for r in json.loads((root / "fits.json").read_text())
            if r["case"] == 0 and r["condition"]["id"] == label
        ]
        if len(matches) != 1:
            raise ValueError("Expected exactly one completed first LoRA fit")
        return matches[0]

    old, new = fit(before), fit(after)
    same_files = {}
    for name in ("inputs.json", "tokenization.json", "case-0-base.json", f"case-0-{label}.json"):
        a, b = (before / name).read_bytes(), (after / name).read_bytes()
        same_files[name] = {
            "exact": a == b,
            "before_sha256": hashlib.sha256(a).hexdigest(),
            "after_sha256": hashlib.sha256(b).hexdigest(),
        }
    arrays = {}
    for name in (
        f"case-0-{label}-teacher.safetensors",
        f"case-0-{label}-adapter/adapter_model.safetensors",
        "case-0-scores.safetensors",
    ):
        a, b = load_file(str(before / name)), load_file(str(after / name))
        if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
            raise ValueError("Different tensor keys or shapes")
        finite = all(torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all() for k in a)
        arrays[name] = {
            "exact": all(torch.equal(a[k], b[k]) for k in a),
            "finite": bool(finite),
            "max_abs_difference": max(
                float((a[k].double() - b[k].double()).abs().max()) for k in a
            ),
        }
    same_history = len(old["history"]) == len(new["history"]) and all(
        all(a[k] == b.get(k) for k in a)
        for a, b in zip(old["history"], new["history"], strict=True)
    )
    result = {
        "condition": label,
        "files": same_files,
        "tensors": arrays,
        "history_exact_ignoring_new_scale_metadata": same_history,
        "metrics_exact": old["metrics"] == new["metrics"],
        "gradient_retries": sum(len(r.get("gradient_retries", [])) for r in new["history"]),
        "semantic_test_outputs_displayed": False,
    }
    result["exact_replay"] = (
        same_history
        and result["metrics_exact"]
        and all(r["exact"] for r in same_files.values())
        and all(r["exact"] and r["finite"] for r in arrays.values())
    )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
