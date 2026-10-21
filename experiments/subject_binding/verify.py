"""Verify frozen execution, exported supports and calibration-only model selection."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from .analysis import select_weight
from .measure import statistics


def read(path):
    return json.loads(path.read_text())


def check_results(root, inputs, protocol, cold, tokenizer=None):
    identity, completion = read(root / "identity.json"), read(root / "completion.json")
    assert (
        identity["protocol"] == protocol and identity["fingerprint"] == protocol["base_fingerprint"]
    )
    assert completion["status"] == "complete" and completion["new_fits"] == 6
    assert completion["weights_restored"] and completion["selection_uses_calibration_only"]
    assert completion["final_fingerprint"] == protocol["base_fingerprint"]
    assert read(root / "inputs.json") == inputs
    expected_cold = [
        {
            "case": c["case"],
            "base": [p["base"] for p in c["probes"]],
            "edited": [p["edited"] for p in c["probes"]],
        }
        for c in cold
    ]
    assert read(root / "cold-checks.json") == expected_cold
    fits = read(root / "fits.json")
    expected = {(c["case"], w) for c in inputs["cases"] for w in protocol["preservation_weights"]}
    assert len(fits) == len(expected) == 6 and {(r["case"], r["weight"]) for r in fits} == expected
    outputs = []
    reconstructed = []
    for case in inputs["cases"]:
        c = case["case"]
        base = read(root / f"case-{c}-base.json")
        reference = load_file(str(root / f"case-{c}-calibration-base.safetensors"))
        assert set(reference) == {"logits", "lengths"}
        lengths = reference["lengths"]
        reference = reference["logits"]
        assert lengths.tolist() == [min(4, len(r["tokens"])) for r in base["calibration"]]
        assert len(lengths) == len(case["calibration"]) == 3 and lengths.dtype == torch.int64
        assert reference.ndim == 2 and reference.shape[0] == int(lengths.sum())
        assert reference.dtype == torch.float32 and torch.isfinite(reference).all()
        for weight in protocol["preservation_weights"]:
            prefix = f"case-{c}-w{int(weight)}"
            record = next(r for r in fits if r["case"] == c and r["weight"] == weight)
            assert record["updates"] == len(record["history"]) == protocol["fit"]["steps"]
            assert [r["step"] for r in record["history"]] == list(range(1, record["updates"] + 1))
            assert record["trainable_scalars"] == case["trainable_scalars"]
            tensors = load_file(str(root / (prefix + "-calibration.safetensors")))
            assert set(tensors) == {"logits", "target_log_probabilities"}
            assert tensors["logits"].shape == reference.shape
            assert (
                tensors["logits"].dtype
                == tensors["target_log_probabilities"].dtype
                == torch.float32
            )
            assert tensors["target_log_probabilities"].shape[0] == len(case["train_prompts"])
            assert tensors["target_log_probabilities"].shape[1] > 0
            assert (tensors["target_log_probabilities"] <= 0).all()
            recomputed = statistics(
                tensors["logits"], reference, tensors["target_log_probabilities"], lengths
            )
            for key in ("nll", "mean_answer_kl", "max_answer_kl"):
                assert math.isclose(
                    recomputed[key], record["final"][key], abs_tol=1e-9, rel_tol=1e-8
                )
            assert recomputed["answer_argmax_retained"] == record["final"]["answer_argmax_retained"]
            reconstructed.append({"case": c, "weight": weight, "final": recomputed})
            artifact = root / (prefix + "-edit")
            manifest = read(artifact / "edit.json")
            delta = load_file(str(artifact / "edit.safetensors"))
            assert manifest["targets"] == case["targets"]
            assert manifest["base_fingerprint"] == protocol["base_fingerprint"]
            assert set(delta) == {"delta"} and delta["delta"].dtype == torch.bfloat16
            assert delta["delta"].shape == (
                len(case["targets"]),
                case["trainable_scalars"] // len(case["targets"]),
            )
            assert delta["delta"].numel() == case["trainable_scalars"]
            assert torch.isfinite(delta["delta"]).all()
            row = read(root / (prefix + ".json"))
            assert row["case"] == c
            for bank, count in (
                ("edit", len(case["edit_probes"])),
                ("utility", len(inputs["utility"])),
                *((k, len(case[k])) for k in ("calibration", "evaluation", "exposed_scope")),
            ):
                assert len(row[bank]) == count
                outputs.extend(row[bank])
        for bank in ("edit", "utility", "calibration", "evaluation", "exposed_scope"):
            assert len(base[bank]) == len(row[bank])
            outputs.extend(base[bank])
    assert len(outputs) == 424
    for row in outputs:
        assert type(row["truncated"]) is bool and isinstance(row["text"], str)
        assert 0 < len(row["tokens"]) <= 40 and all(
            type(t) is int and t >= 0 for t in row["tokens"]
        )
        if tokenizer is not None:
            assert tokenizer.decode(row["tokens"], skip_special_tokens=True) == row["text"]
            assert row["truncated"] == (
                len(row["tokens"]) == 40 and row["tokens"][-1] != tokenizer.eos_token_id
            )
    selected = select_weight(
        reconstructed,
        protocol["preservation_weights"],
        [c["case"] for c in inputs["cases"]],
        protocol["selection"],
    )
    assert selected == completion["selected_weight"]
    return {
        "output_sequences_checked": len(outputs),
        "calibration_records_recomputed": len(reconstructed),
        "selected_weight": selected,
        "cold_sequences_checked": 8,
        "token_decoding_checked": tokenizer is not None,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--executed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).parent
    protocol = read(here / "protocol_v1.json")
    submission = read(here / "submission_v1.json")
    launch = read(here / "launch_v1.json")
    assert (
        hashlib.sha256(args.executed_source.read_bytes()).hexdigest()
        == launch["submitted_source_sha256"]
    )
    assert read(args.download / "submission.json") == submission
    for local, uploaded in (("inputs_v1.json", "inputs.json"), ("cold_v1.json", "cold.json")):
        assert (
            hashlib.sha256((here / local).read_bytes()).hexdigest()
            == protocol["data_sha256"][uploaded]
        )
    assert read(args.download / "provider.json")["returncode"] == 0
    tokenizer = None
    if args.decode:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(
            protocol["model"], revision=protocol["revision"], local_files_only=True
        )
    counts = check_results(
        args.download / "results",
        read(here / "inputs_v1.json"),
        protocol,
        read(here / "cold_v1.json"),
        tokenizer,
    )
    log = (args.download / "execution.log").read_text()
    assert log.count("COLD_VERIFIED ") == 2 and log.count("FIT_RESTORED ") == 6
    assert (
        log.rindex("COLD_VERIFIED ")
        < log.index("STAGE cold_complete_before_new_probes")
        < log.index("FIT_START ")
    )
    assert "COMPLETE subject-binding objective ablation" in log
    result = {
        "status": "Frozen execution and calibration-only selection verified; no quality claim",
        **counts,
        "file_sha256": {
            str(p.relative_to(args.download)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.download.rglob("*"))
            if p.is_file()
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
