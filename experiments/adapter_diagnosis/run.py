"""Replay a frozen successful adapter, then remove complete update groups.

No adapter training, dose search, new correction data or fresh confirmation.
All base/full controls must replay before any new ablation inference.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.adapter_comparison.fit import capture_references
from experiments.fine_transfer.lora_control import (
    gptj_targets,
    load_gptj_lora,
    merge_gptj_lora,
    remove_gptj_lora,
)
from experiments.fine_transfer.run import generate, write
from experiments.subject_binding.measure import measure_merged
from neuronpatch.patch import fingerprint

from .groups import conditions, retain_updates


def verify_parent(parent, expected):
    """Require the completed comparison's exact selected artifacts and outputs."""
    for name, digest in expected.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Parent artifact paths must be relative")
        if hashlib.sha256((parent / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Parent artifact hash differs: {name}")


def locate_parent(search, expected):
    """Kaggle attaches the prior kernel's outputs under /kaggle/input."""
    identity_hash = expected["identity.json"]
    matches = [
        path.parent
        for path in search.rglob("identity.json")
        if hashlib.sha256(path.read_bytes()).hexdigest() == identity_hash
    ]
    if len(matches) != 1:
        raise ValueError("Exactly one attached parent comparison with matching identity required")
    verify_parent(matches[0], expected)
    return matches[0]


def execute(model, tokenizer, parent, protocol, output, check_time, *, resume=None):
    """Run the complete diagnostic on an already-loaded frozen materialized base."""
    from experiments.adapter_comparison.run import restore_weights, snapshot_weights

    verify_parent(parent, protocol["parent_sha256"])
    source_identity = json.loads((parent / "identity.json").read_text())
    completion = json.loads((parent / "completion.json").read_text())
    if (
        completion["status"] != "complete"
        or completion["selected"]["lora"] != protocol["parent_condition"]
        or not completion["weights_restored"]
    ):
        raise ValueError("A completed calibration-selected LoRA comparison is required")
    identity = fingerprint(model)
    assert identity == protocol["base_fingerprint"] == source_identity["fingerprint"]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    names = gptj_targets(model, "attention_mlp")
    variants = conditions(names, len(model.transformer.h))
    assert variants == protocol["conditions"]
    prior_rows, prior, resume_config = [], None, None
    if resume is not None:
        from .verify import check_results

        prior, resume_config = resume
        verify_parent(prior, resume_config["file_sha256"])
        checked = check_results(prior, parent, protocol, partial=True)
        assert checked["remaining_interventions"] == resume_config["remaining_interventions"]
        assert checked["completed_interventions"] == resume_config["completed_interventions"]
        prior_rows = json.loads((prior / "interventions.json").read_text())
    inputs = json.loads((parent / "inputs.json").read_text())
    write(output / "inputs.json", inputs)
    write(output / "identity.json", {"fingerprint": identity, "protocol": protocol})
    snapshot = snapshot_weights(model, names)
    write(
        output / "snapshot.json",
        {"targets": names, "bytes": sum(t.numel() * t.element_size() for t in snapshot.values())},
    )
    contexts, records, controls, base_metrics = {}, [], [], []
    generated_observations = 0

    def evaluate(case):
        nonlocal generated_observations
        result = {"case": case["case"]}
        for bank, probes in [
            ("edit", case["edit_probes"]),
            ("utility", inputs["utility"]),
            *((k, case[k]) for k in ("calibration", "evaluation", "exposed_scope")),
        ]:
            result[bank] = []
            for probe in probes:
                check_time()
                result[bank].append(generate(model, tokenizer, probe["prompt"]))
                generated_observations += 1
        return result

    def intervention(case, variant, refs, reference):
        nonlocal model
        check_time()
        c, name = case["case"], variant["id"]
        print("INTERVENTION_START", c, name, flush=True)
        wrapped = load_gptj_lora(model, parent / f"case-{c}-{protocol['parent_condition']}-edit")
        # Once merged, wrappers are unloaded. Always restore actual base weights,
        # including after a generation failure; never subtract rounded updates.
        try:
            metadata = retain_updates(wrapped, names, variant["keep"])
            model = merge_gptj_lora(wrapped)
            wrapped = None
            metrics, tensors = measure_merged(
                model, tokenizer, case["train_prompts"], case["target"], refs.protected, reference
            )
            generated = evaluate(case)
            if name == "full":
                expected = json.loads(
                    (parent / f"case-{c}-{protocol['parent_condition']}.json").read_text()
                )
                assert generated == expected, "Full merged adapter generation replay differs"
                old = load_file(
                    str(parent / f"case-{c}-{protocol['parent_condition']}-calibration.safetensors")
                )
                assert all(torch.equal(value, old[key]) for key, value in tensors.items()), (
                    "Full merged adapter calibration replay differs"
                )
            save_file(tensors, output / f"case-{c}-{name}-calibration.safetensors")
            write(output / f"case-{c}-{name}.json", generated)
        finally:
            if wrapped is not None:
                model = remove_gptj_lora(wrapped)
            restore_weights(model, snapshot)
        assert fingerprint(model) == identity
        row = {"case": c, "condition": name, "metrics": metrics, **metadata, "base_restored": True}
        records.append(row)
        write(output / "interventions.json", records)
        print("INTERVENTION_RESTORED", c, name, metrics, flush=True)

    # Both cases' full base/adapter continuations and calibration tensors replay
    # before any partial adapter is evaluated. A mismatch stops the worker.
    for case in inputs["cases"]:
        c = case["case"]
        base = evaluate(case)
        assert base == json.loads((parent / f"case-{c}-base.json").read_text()), (
            "Base generation replay differs"
        )
        write(output / f"case-{c}-base.json", base)
        refs = capture_references(
            model,
            tokenizer,
            case["train_prompts"],
            case["target"],
            [p["prompt"] for p in case["calibration"]],
        )
        reference = load_file(str(parent / f"case-{c}-calibration-base.safetensors"))
        actual = torch.cat([p[2].cpu() for p in refs.protected])
        assert torch.equal(actual, reference["logits"]), "Base calibration replay differs"
        save_file(reference, output / f"case-{c}-calibration-base.safetensors")
        metrics, tensors = measure_merged(
            model, tokenizer, case["train_prompts"], case["target"], refs.protected, reference
        )
        save_file(tensors, output / f"case-{c}-base-calibration.safetensors")
        base_metrics.append({"case": c, "metrics": metrics})
        write(output / "base-metrics.json", base_metrics)
        contexts[c] = refs, reference
        intervention(case, variants[0], refs, reference)
        controls.append(
            {
                "case": c,
                "base_outputs_exact": True,
                "full_outputs_exact": True,
                "base_calibration_exact": True,
                "full_calibration_exact": True,
            }
        )
        write(output / "controls.json", controls)
    print("STAGE all_controls_replayed_before_ablation", flush=True)
    copied = []
    for row in prior_rows:
        if row["condition"] == "full":
            continue  # Full controls have just been regenerated and exactly replayed.
        c, name = row["case"], row["condition"]
        for suffix in (".json", "-calibration.safetensors"):
            filename = f"case-{c}-{name}{suffix}"
            assert not (output / filename).exists()
            shutil.copyfile(prior / filename, output / filename)
        records.append(row)
        copied.append({"case": c, "condition": name})
    if resume is not None:
        write(output / "interventions.json", records)
        write(
            output / "resume.json",
            {
                "source": resume_config,
                "copied_interventions": copied,
                "full_controls_regenerated": True,
                "remaining_interventions": resume_config["remaining_interventions"],
            },
        )
        print("STAGE verified_partial_results_copied", len(copied), flush=True)
    completed = {(r["case"], r["condition"]) for r in records}
    for case in inputs["cases"]:
        refs, reference = contexts[case["case"]]
        for variant in variants[1:]:
            if (case["case"], variant["id"]) in completed:
                continue
            intervention(case, variant, refs, reference)
    assert len(records) == len(inputs["cases"]) * len(variants)
    assert fingerprint(model) == identity
    write(
        output / "completion.json",
        {
            "status": "complete",
            "new_fits": 0,
            "new_training_steps": 0,
            "interventions": len(records),
            "controls_exact": True,
            "weights_restored": True,
            "final_fingerprint": identity,
            "resumed": resume is not None,
            "copied_interventions": len(copied),
            "new_interventions_this_worker": len(records) - len(copied),
            "new_observed_continuations": generated_observations,
        },
    )
    return records


def main():
    from transformers import GPT2Tokenizer, GPTJForCausalLM

    from experiments.adapter_comparison.run import dependency_preflight

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    dependency_preflight(protocol["runtime"])
    parent = args.parent or locate_parent(Path("/kaggle/input"), protocol["parent_sha256"])
    verify_parent(parent, protocol["parent_sha256"])
    resume = None
    resume_file = args.data / "resume.json"
    if resume_file.exists():
        resume_config = json.loads(resume_file.read_text())
        prior = locate_parent(Path("/kaggle/input"), resume_config["file_sha256"])
        resume = prior, resume_config
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Frozen-adapter diagnostic exceeded runtime limit")

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
    execute(model, tokenizer, parent, protocol, args.output, check_time, resume=resume)
    print("COMPLETE adapter-diagnosis", flush=True)


if __name__ == "__main__":
    main()
