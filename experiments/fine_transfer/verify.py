"""Independently check downloaded v1 outputs before interpreting pilot scores."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def read(path):
    return json.loads(path.read_text())


def verify(directory, tokenizer=None):
    """Verify run identity, complete paired comparisons and compact artifact shapes.

    directory is the downloaded transfer-results folder, containing submission,
    provider and results. Checks use the committed frozen v1 specification.
    This proves output integrity, not model-answer correctness or method quality.
    """
    here = Path(__file__).parent
    protocol = read(here / "protocol_v1.json")
    inputs = read(here / "inputs_v1.json")
    expected_submission = read(here / "submission_v1.json")
    if read(directory / "submission.json") != expected_submission:
        raise ValueError("Executed submission differs from frozen source/data hashes")
    assert (
        hashlib.sha256((here / "inputs_v1.json").read_bytes()).hexdigest()
        == protocol["inputs_sha256"]
    )
    provider = read(directory / "provider.json")
    assert provider["returncode"] == 0
    root = directory / "results"
    identity = read(root / "identity.json")
    completion = read(root / "completion.json")
    assert identity["protocol"] == protocol
    assert completion["status"] == "complete" and completion["weights_restored"]
    assert (
        identity["fingerprint"] == completion["final_fingerprint"] == protocol["base_fingerprint"]
    )
    conditions = (
        "fine_augmented",
        "gradient_single",
        "gradient_consensus",
        "consensus_layer_matched_random",
    )
    expected_fits = {(r["case"], name) for r in inputs["cases"] for name in conditions}
    actual_fits = [(r["case"], r["condition"]) for r in completion["fits"]]
    assert len(actual_fits) == len(expected_fits) == 8 and set(actual_fits) == expected_fits
    utility = read(root / "base-utility.json")
    assert len(utility) == len(inputs["utility"]) == 32
    for actual, expected in zip(utility, inputs["utility"], strict=True):
        assert all(actual[k] == value for k, value in expected.items())
    outputs = list(utility)
    fit_geometry = []
    for record in inputs["cases"]:
        case = record["case"]
        manifest_path = (
            here.parent
            / "fine_reference/library_parity_v1/artifact_metadata"
            / f"case-{case:03d}-random-False-edit.json"
        )
        assert (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            == protocol["artifact_sha256"][f"case-{case}/edit.json"]
        )
        occurrences = len(read(manifest_path)["targets"])
        selected_targets = {}
        reference = read(root / f"case-{case}-fine-original.json")
        assert reference["probes"] == record["probes"]
        assert [
            r["text"] for r in reference["after"][: len(record["reference_outputs"])]
        ] == record["reference_outputs"]
        before = reference["before"]
        for name in ("fine-original", *conditions):
            row = read(root / f"case-{case}-{name}.json")
            assert row["case"] == case and row["probes"] == record["probes"]
            assert row["before"] == before
            assert len(row["after"]) == len(before) == len(record["probes"])
            assert len(row["utility"]) == len(utility)
            outputs += row["after"] + row["utility"]
            if name == "fine-original":
                continue
            assert row["condition"] == name
            assert len(row["targets"]) == occurrences
            assert row["trainable_scalars"] == occurrences * 4096
            selected_targets[name] = row["targets"]
            assert 0 <= row["updates"] <= 50
            artifact = root / f"case-{case}-{name}-edit"
            manifest = read(artifact / "edit.json")
            assert manifest["base_fingerprint"] == identity["fingerprint"]
            assert manifest["targets"] == row["targets"]
            assert manifest["metadata"]["condition"] == name
            tensors = load_file(str(artifact / "edit.safetensors"))
            assert set(tensors) == {"delta"}
            delta = tensors["delta"]
            assert delta.dtype == torch.bfloat16 and torch.isfinite(delta).all()
            assert delta.shape == (len(row["targets"]), 4096)
            summed = {}
            for target, vector in zip(row["targets"], delta.double(), strict=True):
                key = tuple(target)
                summed[key] = summed.get(key, torch.zeros_like(vector)) + vector
            fit_geometry.append(
                {
                    "case": case,
                    "condition": name,
                    "occurrences": occurrences,
                    "unique_columns": len(summed),
                    "modules": len({key[0] for key in summed}),
                    "parameter_delta_norm": float(delta.double().norm()),
                    "linearly_summed_column_delta_norm": float(
                        torch.stack(list(summed.values())).norm()
                    ),
                    "caution": (
                        "Summed deltas are a linear diagnostic, "
                        "not actual FP16 sequential merged weights."
                    ),
                }
            )
        candidate = selected_targets["gradient_consensus"]
        random = selected_targets["consensus_layer_matched_random"]
        assert [row[0] for row in candidate] == [row[0] for row in random]
        for i in range(occurrences):
            for j in range(occurrences):
                assert (candidate[i] == candidate[j]) == (random[i] == random[j])
        outputs += before
    for row in outputs:
        assert 0 < len(row["tokens"]) <= 40
        if tokenizer is not None:
            assert tokenizer.decode(row["tokens"], skip_special_tokens=True) == row["text"]
            assert row["truncated"] == (
                len(row["tokens"]) == 40 and row["tokens"][-1] != tokenizer.eos_token_id
            )
    files = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    return {
        "status": "Downloaded frozen-v1 comparison verified; no quality claim",
        "new_fits": len(actual_fits),
        "reference_edits": len(inputs["cases"]),
        "output_sequences_checked": len(outputs),
        "token_decoding_checked": tokenizer is not None,
        "fit_geometry": fit_geometry,
        "file_sha256": files,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true", help="Use cached pinned tokenizer")
    args = parser.parse_args()
    tokenizer = None
    if args.decode:
        from transformers import GPT2Tokenizer

        protocol = read(Path(__file__).parent / "protocol_v1.json")
        tokenizer = GPT2Tokenizer.from_pretrained(
            protocol["model"], revision=protocol["revision"], local_files_only=True
        )
    result = verify(args.download, tokenizer)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
