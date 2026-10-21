"""Independently recompute calibration, verify artifacts, then permit scoring."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.subject_binding.measure import statistics

from .analysis import choose_rates


def read(path):
    return json.loads(path.read_text())


def metrics(tensors, reference, lengths, prompts, tokenizer, target):
    assert set(tensors) == {"logits", "target_log_probabilities"}
    assert tensors["logits"].shape == reference.shape
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
    assert tensors["target_log_probabilities"].shape[0] == len(prompts)
    assert tensors["target_log_probabilities"].shape[1] > 0
    assert (tensors["target_log_probabilities"] <= 0).all()
    if tokenizer is not None:
        assert tensors["target_log_probabilities"].shape[1] == len(
            tokenizer.encode(" " + target, add_special_tokens=False)
        )
    return statistics(tensors["logits"], reference, tensors["target_log_probabilities"], lengths)


def compare_metrics(actual, reported):
    for key in ("nll", "mean_answer_kl", "max_answer_kl"):
        assert math.isclose(actual[key], reported[key], abs_tol=1e-9, rel_tol=1e-8)
    assert actual["answer_argmax_retained"] == reported["answer_argmax_retained"]


def check_results(root, inputs, protocol, cold, tokenizer=None):
    identity, completion = read(root / "identity.json"), read(root / "completion.json")
    assert (
        identity["protocol"] == protocol and identity["fingerprint"] == protocol["base_fingerprint"]
    )
    assert identity["torch"].split("+")[0] == protocol["runtime"]["torch"]
    assert all(identity["runtime"][n].split("+")[0] == v for n, v in protocol["runtime"].items())
    assert completion["status"] == "complete" and completion["new_fits"] == protocol["new_fits"]
    assert completion["weights_restored"] and completion["selection_uses_calibration_only"]
    assert completion["final_fingerprint"] == protocol["base_fingerprint"]
    assert read(root / "inputs.json") == inputs
    assert read(root / "cold-checks.json") == [
        {
            "case": c["case"],
            "base": [p["base"] for p in c["probes"]],
            "edited": [p["edited"] for p in c["probes"]],
        }
        for c in cold
    ]
    fits = read(root / "fits.json")
    dimensions = protocol["model_dimensions"]
    expected_names = {
        f"transformer.h.{i}.{suffix}"
        for i in range(dimensions["layers"])
        for suffix in [
            *(f"attn.{p}_proj" for p in ("q", "k", "v", "out")),
            "mlp.fc_in",
            "mlp.fc_out",
        ]
    }
    assert set(read(root / "snapshot.json")["targets"]) == expected_names
    # Validate all fit identities before constructing reconstructed metrics.
    choose_rates(
        fits, protocol["conditions"], [c["case"] for c in inputs["cases"]], protocol["selection"]
    )
    assert len(fits) == protocol["new_fits"]
    output_count, decoded, reconstructed = 0, 0, []
    for case in inputs["cases"]:
        c = case["case"]
        base = read(root / f"case-{c}-base.json")
        refs = load_file(str(root / f"case-{c}-calibration-base.safetensors"))
        assert set(refs) == {"logits", "lengths"}
        reference, lengths = refs["logits"], refs["lengths"]
        assert lengths.tolist() == [min(4, len(r["tokens"])) for r in base["calibration"]]
        assert len(lengths) == len(case["calibration"]) == 3 and lengths.dtype == torch.int64
        assert reference.dtype == torch.float32 and torch.isfinite(reference).all()
        assert reference.ndim == 2 and len(reference) == int(lengths.sum())
        selection = read(root / f"case-{c}-selection.json")
        mean_support = selection["supports"]["mean_gradient"]
        assert len(mean_support) == len(case["targets"])
        outputs = [base]
        for condition in protocol["conditions"]:
            name = condition["id"]
            row = next(r for r in fits if r["case"] == c and r["condition"] == name)
            assert row["family"] == condition["family"] and row["lr"] == condition["lr"]
            assert row["updates"] == len(row["history"]) == protocol["fit"]["steps"]
            assert [r["step"] for r in row["history"]] == list(range(1, row["updates"] + 1))
            assert row["base_restored"]
            budget = condition["expected_scalars"][str(c)]
            assert row["metadata"]["trainable_scalars"] == budget
            prefix = f"case-{c}-{name}"
            tensors = load_file(str(root / (prefix + "-calibration.safetensors")))
            final = metrics(
                tensors, reference, lengths, case["train_prompts"], tokenizer, case["target"]
            )
            compare_metrics(final, row["final"])
            reconstructed.append({"case": c, "condition": name, "final": final})
            live_tensors = load_file(str(root / (prefix + "-live.safetensors")))
            live = metrics(
                live_tensors, reference, lengths, case["train_prompts"], tokenizer, case["target"]
            )
            compare_metrics(live, row["live_final"])
            artifact = root / (prefix + "-edit")
            if condition["family"] == "lora":
                assert row["roundtrip"] == {
                    "factors_exact": True,
                    "logits_exact": True,
                    "base_unchanged": True,
                }
                reloaded = load_file(str(root / (prefix + "-reloaded.safetensors")))
                assert reloaded.keys() == live_tensors.keys()
                assert all(torch.equal(t, reloaded[n]) for n, t in live_tensors.items())
                config = read(artifact / "adapter_config.json")
                assert config["r"] == config["lora_alpha"] == protocol["lora_rank"]
                assert config["lora_dropout"] == 0 and config["bias"] == "none"
                assert not config["use_dora"] and not config["use_rslora"]
                assert not config["fan_in_fan_out"] and not config["modules_to_save"]
                factors = load_file(str(artifact / "adapter_model.safetensors"))
                names = row["metadata"]["targets"]
                assert set(names) == expected_names and len(names) == len(expected_names)
                assert set(config["target_modules"]) == expected_names or all(
                    any(
                        n == suffix or n.endswith("." + suffix)
                        for suffix in config["target_modules"]
                    )
                    for n in names
                )
                keys = {
                    f"base_model.model.{n}.lora_{part}.weight" for n in names for part in ("A", "B")
                }
                assert factors.keys() == keys
                rank, width, inner = (
                    protocol["lora_rank"],
                    dimensions["width"],
                    dimensions["mlp_width"],
                )
                for target in names:
                    incoming = inner if target.endswith("mlp.fc_out") else width
                    outgoing = inner if target.endswith("mlp.fc_in") else width
                    assert factors[f"base_model.model.{target}.lora_A.weight"].shape == (
                        rank,
                        incoming,
                    )
                    assert factors[f"base_model.model.{target}.lora_B.weight"].shape == (
                        outgoing,
                        rank,
                    )
                assert sum(t.numel() for t in factors.values()) == budget
                assert all(
                    t.dtype == torch.float32 and torch.isfinite(t).all() for t in factors.values()
                )
            else:
                manifest = read(artifact / "edit.json")
                support = case["targets"] if condition["family"] == "consensus" else mean_support
                assert manifest["targets"] == support
                assert manifest["base_fingerprint"] == protocol["base_fingerprint"]
                assert manifest["delta_dtype"] == "float32"
                delta = load_file(str(artifact / "edit.safetensors"))
                assert set(delta) == {"delta"} and delta["delta"].dtype == torch.float32
                assert delta["delta"].shape == (len(support), dimensions["width"])
                assert delta["delta"].shape[0] == len(support) and delta["delta"].numel() == budget
                assert torch.isfinite(delta["delta"]).all()
            output = read(root / (prefix + ".json"))
            assert output["case"] == c
            outputs.append(output)
        for output in outputs:
            for bank, count in [
                ("edit", len(case["edit_probes"])),
                ("utility", len(inputs["utility"])),
            ] + [
                (name, len(case[name])) for name in ("calibration", "evaluation", "exposed_scope")
            ]:
                assert len(output[bank]) == count
                for generation in output[bank]:
                    output_count += 1
                    assert (
                        isinstance(generation["text"], str)
                        and type(generation["truncated"]) is bool
                    )
                    tokens = generation["tokens"]
                    assert 0 < len(tokens) <= 40 and all(type(t) is int and t >= 0 for t in tokens)
                    if tokenizer is not None:
                        assert (
                            tokenizer.decode(tokens, skip_special_tokens=True) == generation["text"]
                        )
                        assert generation["truncated"] == (
                            len(tokens) == 40 and tokens[-1] != tokenizer.eos_token_id
                        )
                        decoded += 1
    selected = choose_rates(
        reconstructed,
        protocol["conditions"],
        [c["case"] for c in inputs["cases"]],
        protocol["selection"],
    )
    assert selected == completion["selected"]
    return {
        "sequences_checked": output_count,
        "decoded": decoded,
        "calibration_records_recomputed": len(reconstructed),
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--executed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).parent
    protocol, launch, submission = [
        read(here / name) for name in ("protocol_v1.json", "launch_v1.json", "submission_v1.json")
    ]
    assert (
        hashlib.sha256(args.executed_source.read_bytes()).hexdigest()
        == launch["submitted_source_sha256"]
    )
    assert read(args.download / "submission.json") == submission
    for name, local in (("inputs.json", "inputs_v1.json"), ("cold.json", "cold_v1.json")):
        assert (
            hashlib.sha256((here.parent / "subject_binding" / local).read_bytes()).hexdigest()
            == protocol["data_sha256"][name]
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
        read(here.parent / "subject_binding/inputs_v1.json"),
        protocol,
        read(here.parent / "subject_binding/cold_v1.json"),
        tokenizer,
    )
    log = (args.download / "execution.log").read_text()
    assert log.count("FIT_RESTORED ") == protocol["new_fits"]
    assert (
        log.rindex("COLD_VERIFIED ")
        < log.index("STAGE cold_complete_before_new_inference")
        < log.index("FIT_START ")
    )
    assert "COMPLETE adapter-comparison" in log
    result = {
        "status": "Execution and calibration independently verified; no quality claim",
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
