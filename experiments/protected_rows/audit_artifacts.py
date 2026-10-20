"""Check supports, training histories and exported tensors from a completed pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution = read(args.results / "manifest.json")
    if not execution.get("completed"):
        raise ValueError("A completed execution is required")
    protocol = execution["protocol"]
    selected = {name: row["indices"] for name, row in read(args.results / "selection.json").items()}
    random_rows = read(args.results / "random_rows.json")
    expected_targets = {f"model.layers.{layer}.mlp.up_proj" for layer in protocol["layers"]}
    assert selected.keys() == random_rows.keys() == expected_targets
    for mapping in (selected, random_rows):
        for indices in mapping.values():
            assert len(indices) == protocol["rows_per_layer"]
            assert sorted(set(indices)) == indices and 0 <= min(indices) <= max(indices) < 9728
    base_cal = {row["id"]: row for row in read(args.results / "base_cal.json")}
    anchors = read(args.results / "anchors.json")
    assert protocol["minimum_anchors"] <= len(anchors) <= protocol["anchor_count"]
    assert len({a["article"] for a in anchors}) == len(anchors)
    assert all(a["_target_ids"] == base_cal[a["id"]]["generated_token_ids"] for a in anchors)
    assert all(base_cal[a["id"]]["em"] == 1 and a["kind"] == "answerable" for a in anchors)
    fingerprints, output = set(), {}
    first_losses = []
    for config in protocol["fits"]:
        name, family = config["name"], config["family"]
        directory = args.results / name
        effective_config = read(directory / "config.json")
        assert all(effective_config[key] == value for key, value in config.items())
        assert effective_config["protocol"] == protocol
        history = read(directory / "history.json")
        assert [row["step"] for row in history] == list(range(1, protocol["training_examples"] + 1))
        assert not any(row["optimizer_skipped"] for row in history)
        first_losses.append((history[0]["correction_ce"], history[0]["preservation_ce"]))
        for row in history:
            assert all(
                torch.isfinite(torch.tensor(row[key]))
                for key in ("correction_ce", "preservation_ce", "loss_scale")
            )
        for form in ("live", "merged"):
            predictions = read(directory / f"anchors_{form}.json")
            assert {row["id"] for row in predictions} == {a["id"] for a in anchors}
            for row in predictions:
                assert row["text"] == base_cal[row["id"]]["text"]
                assert row["generated_token_ids"] == base_cal[row["id"]]["generated_token_ids"]
        result = {"steps": len(history), "optimizer_skips": 0, "anchor_trajectories": "exact"}
        if family != "lora":
            metadata = read(directory / "patch/patch.json")
            assert metadata["format_version"] == 2 and metadata["method"] == "linear_output_rows"
            expected = random_rows if family == "random" else selected
            assert metadata["targets"] == expected == effective_config["indices"]
            assert metadata["metadata"]["protocol"] == protocol
            assert all(metadata["metadata"][key] == value for key, value in config.items())
            fingerprints.add(metadata["base_fingerprint"])
            tensors = load_file(directory / "patch/patch.safetensors")
            assert tensors.keys() == expected.keys()
            for target, tensor in tensors.items():
                assert tuple(tensor.shape) == (len(expected[target]), protocol["hidden_size"])
                assert tensor.dtype == torch.float16 and torch.isfinite(tensor).all()
            result["delta_frobenius_norm"] = (
                sum(tensor.double().square().sum().item() for tensor in tensors.values()) ** 0.5
            )
        else:
            metadata = read(directory / "adapter/adapter_config.json")
            assert metadata["r"] == protocol["lora_rank"]
            assert metadata["lora_alpha"] == protocol["lora_alpha"]
            assert metadata["bias"] == "none"
            # PEFT compresses explicit names to an equivalent suffix when possible.
            full_targets = {f"model.layers.{layer}.mlp.up_proj" for layer in range(36)}
            assert set(metadata["target_modules"]) in ({"up_proj"}, full_targets)
            tensors = load_file(directory / "adapter/adapter_model.safetensors")
            assert len(tensors) == 72
            for layer in range(36):
                prefix = f"base_model.model.model.layers.{layer}.mlp.up_proj"
                a, b = (tensors[f"{prefix}.lora_{factor}.weight"] for factor in ("A", "B"))
                assert tuple(a.shape) == (protocol["lora_rank"], protocol["hidden_size"])
                assert tuple(b.shape) == (9728, protocol["lora_rank"])
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
        result["saved_scalars"] = sum(tensor.numel() for tensor in tensors.values())
        assert result["saved_scalars"] == execution["fits"][name]["allocated_parameters"]
        output[name] = result
    assert len(set(first_losses)) == 1
    assert len(fingerprints) == 1
    low = read(args.results / "lora-0/history.json")[0]
    high = read(args.results / "lora-1/history.json")[0]
    assert low["gradient_norm"] == high["gradient_norm"]
    record = {
        "status": "passed",
        "scope": "Saved-artifact consistency; not an independent model forward-pass reproduction",
        "row_base_fingerprint": next(iter(fingerprints)),
        "all_first_step_losses_exact": first_losses[0],
        "lora_first_step_gradient_norm_exact": low["gradient_norm"],
        "lora_target_metadata": "PEFT saved common up_proj suffix; all 36 tensor paths verified",
        "peft_warning": (
            "Second LoRA fit warns about a retained peft_config attribute. It has the expected "
            "72 saved factor tensors and 442368 scalars, with identical initial losses and "
            "gradient norm. Source overwrites the same default config; no extra adapter "
            "parameters are present in the saved artifact."
        ),
        "fits": output,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
