"""One bounded comparison under a common prediction-space training objective."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import GPT2Tokenizer, GPTJForCausalLM

from experiments.fine_transfer.fit import sequence
from experiments.fine_transfer.gradient_controls import discover_controls
from experiments.fine_transfer.lora_control import (
    attach_gptj_lora,
    gptj_targets,
    load_gptj_lora,
    merge_gptj_lora,
    remove_gptj_lora,
)
from experiments.fine_transfer.run import generate, merge, restore, save_columns, write
from experiments.fine_transfer.scope_analysis import digest_output
from experiments.subject_binding.measure import measure_merged
from neuronpatch.occurrence import OccurrenceColumnEdit
from neuronpatch.patch import fingerprint

from .analysis import choose_rates
from .fit import capture_references, fit_shared


@torch.no_grad()
def snapshot_weights(model, names):
    """Keep one CPU snapshot of mergeable weights, never a second GPU model."""
    if not names or len(set(names)) != len(names):
        raise ValueError("Unique nonempty target names required")
    return {name: model.get_submodule(name).weight.detach().cpu().clone() for name in names}


@torch.no_grad()
def restore_weights(model, snapshot):
    for name, reference in snapshot.items():
        weight = model.get_submodule(name).weight
        assert weight.shape == reference.shape and weight.dtype == reference.dtype
        weight.copy_(reference)


def factor_snapshot(model):
    return {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if "lora_" in name
    }


def dependency_preflight(expected):
    """Import the full adapter stack before downloading or allocating the model.

    Old PEFT imports torch's extension loader, which needs setuptools even for
    ordinary LoRA. A seeded developer environment can hide that dependency.
    """
    runtime = {name: importlib.metadata.version(name) for name in expected}
    if not all(runtime[n].split("+")[0] == value for n, value in expected.items()):
        raise RuntimeError("Installed runtime differs from the frozen protocol")
    importlib.import_module("setuptools")
    importlib.import_module("peft")
    print("STAGE dependency_preflight_complete", runtime, flush=True)
    return runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.data / "protocol.json").read_text())
    runtime = dependency_preflight(protocol["runtime"])
    for name, expected in protocol["data_sha256"].items():
        assert hashlib.sha256((args.data / name).read_bytes()).hexdigest() == expected, name
    inputs = json.loads((args.data / "inputs.json").read_text())
    cold = json.loads((args.data / "cold.json").read_text())
    write(args.output / "inputs.json", inputs)
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded adapter comparison exceeded runtime limit")

    print("STAGE model_load", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = (
        GPTJForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        .eval()
        .requires_grad_(False)
    )
    identity = fingerprint(model)
    assert identity == protocol["base_fingerprint"]
    write(
        args.output / "identity.json",
        {
            "fingerprint": identity,
            "protocol": protocol,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "runtime": runtime,
        },
    )
    cold_checks = []
    for row in cold:
        c = row["case"]
        check_time()
        before = [digest_output(generate(model, tokenizer, p["prompt"])) for p in row["probes"]]
        assert before == [p["base"] for p in row["probes"]]
        edit = OccurrenceColumnEdit.load(model, args.data / f"case-{c}-reference")
        targets, delta = list(edit.targets), edit.delta.detach().cpu().clone()
        del edit
        saved = save_columns(model, targets)
        try:
            merge(model, targets, delta)
            actual = [digest_output(generate(model, tokenizer, p["prompt"])) for p in row["probes"]]
            assert actual == [p["edited"] for p in row["probes"]]
        finally:
            restore(model, saved)
        assert digest_output(generate(model, tokenizer, row["probes"][0]["prompt"])) == before[0]
        cold_checks.append({"case": c, "base": before, "edited": actual})
        write(args.output / "cold-checks.json", cold_checks)
        print("COLD_VERIFIED", c, flush=True)
        del delta, saved
    assert fingerprint(model) == identity
    print("STAGE cold_complete_before_new_inference", flush=True)
    broad_names = gptj_targets(model, "attention_mlp")
    snapshot = snapshot_weights(model, broad_names)
    write(
        args.output / "snapshot.json",
        {
            "storage": "CPU",
            "bytes": sum(t.numel() * t.element_size() for t in snapshot.values()),
            "targets": broad_names,
        },
    )
    fits = []
    for case in inputs["cases"]:
        c = case["case"]

        def evaluate():
            output = {"case": c}
            for bank, probes in [("edit", case["edit_probes"]), ("utility", inputs["utility"])] + [
                (name, case[name]) for name in ("calibration", "evaluation", "exposed_scope")
            ]:
                output[bank] = []
                for probe in probes:
                    check_time()
                    output[bank].append(generate(model, tokenizer, probe["prompt"]))
            return output

        base = evaluate()
        write(args.output / f"case-{c}-base.json", base)
        refs = capture_references(
            model,
            tokenizer,
            case["train_prompts"],
            case["target"],
            [p["prompt"] for p in case["calibration"]],
        )
        reference = {
            "logits": torch.cat([p[2].cpu() for p in refs.protected]),
            "lengths": torch.tensor([len(p[2]) for p in refs.protected]),
        }
        save_file(reference, args.output / f"case-{c}-calibration-base.safetensors")
        names = [
            f"transformer.h.{i}.mlp.fc_out"
            for i in range(len(model.transformer.h) - protocol["selection_layers_excluded"])
        ]
        discovery = discover_controls(
            model,
            [sequence(tokenizer, p, case["target"], model.device) for p in case["train_prompts"]],
            names,
            neurons_per_token=protocol["neurons_per_token"],
        )
        discovery["recomputed_worst_matches_saved"] = discovery["supports"]["worst_gradient"] == [
            tuple(t) for t in case["targets"]
        ]
        write(args.output / f"case-{c}-selection.json", discovery)
        supports = {
            "consensus": [tuple(t) for t in case["targets"]],
            "mean": discovery["supports"]["mean_gradient"],
        }
        for condition in protocol["conditions"]:
            check_time()
            name, family = condition["id"], condition["family"]
            print("FIT_START", c, name, flush=True)
            stamp = time.monotonic()
            if family == "lora":
                fitted, metadata = attach_gptj_lora(
                    model,
                    rank=protocol["lora_rank"],
                    scope="attention_mlp",
                    seed=protocol["seed"],
                )
                metadata["status"] = (
                    "Fitted under the registered common-objective development protocol"
                )
                parameters = [p for p in fitted.parameters() if p.requires_grad]
            else:
                fitted = OccurrenceColumnEdit(
                    model, supports[family], delta_dtype=torch.float32
                ).eval()
                parameters = list(fitted.trainable_parameters())
                metadata = {"targets": fitted.targets, "trainable_dtype": "float32"}
            scalars = sum(p.numel() for p in parameters)
            metadata["trainable_scalars"] = scalars
            assert scalars == condition["expected_scalars"][str(c)]
            history = fit_shared(
                fitted,
                parameters,
                refs,
                lr=condition["lr"],
                check_time=check_time,
                **protocol["fit"],
            )
            del parameters
            with nullcontext(fitted) if family == "lora" else fitted.applied() as measured:
                live, live_tensors = measure_merged(
                    measured,
                    tokenizer,
                    case["train_prompts"],
                    case["target"],
                    refs.protected,
                    reference,
                )
            save_file(live_tensors, args.output / f"case-{c}-{name}-live.safetensors")
            artifact = args.output / f"case-{c}-{name}-edit"
            roundtrip = None
            if family == "lora":
                fitted.save_pretrained(
                    artifact, safe_serialization=True, save_embedding_layers=False
                )
                factors = factor_snapshot(fitted)
                model = remove_gptj_lora(fitted)
                del fitted
                assert fingerprint(model) == identity
                fitted = load_gptj_lora(model, artifact)
                reloaded_factors = factor_snapshot(fitted)
                assert factors.keys() == reloaded_factors.keys()
                assert all(torch.equal(t, reloaded_factors[n]) for n, t in factors.items())
                reloaded_metrics, reloaded = measure_merged(
                    fitted,
                    tokenizer,
                    case["train_prompts"],
                    case["target"],
                    refs.protected,
                    reference,
                )
                assert all(torch.equal(t, reloaded[n]) for n, t in live_tensors.items())
                assert reloaded_metrics == live
                save_file(reloaded, args.output / f"case-{c}-{name}-reloaded.safetensors")
                roundtrip = {"factors_exact": True, "logits_exact": True, "base_unchanged": True}
                del factors, reloaded_factors, reloaded
                try:
                    model = merge_gptj_lora(fitted)
                    del fitted
                    final, tensors = measure_merged(
                        model,
                        tokenizer,
                        case["train_prompts"],
                        case["target"],
                        refs.protected,
                        reference,
                    )
                    output = evaluate()
                finally:
                    restore_weights(model, snapshot)
            else:
                fitted.save(artifact, metadata={"condition": name, "case": c, **metadata})
                delta = fitted.delta.detach().cpu().clone()
                targets = list(fitted.targets)
                del fitted
                saved = save_columns(model, targets)
                try:
                    merge(model, targets, delta)
                    final, tensors = measure_merged(
                        model,
                        tokenizer,
                        case["train_prompts"],
                        case["target"],
                        refs.protected,
                        reference,
                    )
                    output = evaluate()
                finally:
                    restore(model, saved)
                del delta, saved
            save_file(tensors, args.output / f"case-{c}-{name}-calibration.safetensors")
            write(args.output / f"case-{c}-{name}.json", output)
            assert fingerprint(model) == identity
            assert generate(model, tokenizer, case["edit_probes"][0]["prompt"]) == base["edit"][0]
            fits.append(
                {
                    "case": c,
                    "condition": name,
                    "family": family,
                    "lr": condition["lr"],
                    "updates": len(history),
                    "history": history,
                    "live_final": live,
                    "final": final,
                    "metadata": metadata,
                    "roundtrip": roundtrip,
                    "base_restored": True,
                    "seconds": time.monotonic() - stamp,
                }
            )
            write(args.output / "fits.json", fits)
            print("FIT_RESTORED", c, name, "final", final, flush=True)
            del output, tensors, live_tensors
            gc.collect()
            torch.cuda.empty_cache()
    assert len(fits) == protocol["new_fits"]
    selected = choose_rates(
        fits,
        protocol["conditions"],
        [case["case"] for case in inputs["cases"]],
        protocol["selection"],
    )
    assert fingerprint(model) == identity
    write(
        args.output / "completion.json",
        {
            "status": "complete",
            "new_fits": len(fits),
            "selected": selected,
            "selection_uses_calibration_only": True,
            "weights_restored": True,
            "final_fingerprint": identity,
            "seconds": time.monotonic() - started,
        },
    )
    print("COMPLETE adapter-comparison", flush=True)


if __name__ == "__main__":
    main()
