"""Reproduce seven frozen50-step fits while recording earlier adequate checkpoints."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import (
    attach_lora,
    layout,
    load_lora,
    merge_lora,
    supports,
    unload_lora,
)
from experiments.paired_mlp.run import generate, write
from experiments.paired_mlp.selection import capture_response_factors, projection_scores
from experiments.paired_mlp.training import fit, teacher, teacher_metrics
from neuronpatch.paired import MLPSelection, PairedMLPPatch
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .geometry import nominal_update_norm
from .observe import Observer, calibration
from .prepare import sha256


def verify_parents(parents, protocol):
    for group, manifest in protocol["parent_manifests"].items():
        for name, digest in manifest.items():
            assert sha256(parents[group] / name) == digest, f"Changed{group}/{name}"


def find_parents(root, protocol):
    found = {}
    identities = list(root.rglob("identity.json"))
    for group, manifest in protocol["parent_manifests"].items():
        paths = [p.parent for p in identities if sha256(p) == manifest["identity.json"]]
        assert len(paths) == 1, (group, len(paths))
        found[group] = paths[0]
    verify_parents(found, protocol)
    return found


def _selections(serialized):
    return [
        MLPSelection(tuple(r["incoming"]), r["outgoing"], tuple(r["indices"])) for r in serialized
    ]


def _serialized(selections):
    return [
        {"incoming": list(s.incoming), "outgoing": s.outgoing, "indices": list(s.indices)}
        for s in selections
    ]


def _same_tensors(left, right):
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


def execute(model, tokenizer, parents, protocol, output, check_time):
    verify_parents(parents, protocol)
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    identity = fingerprint(model)
    assert identity == protocol["base_fingerprint"]
    groups, counts, budgets = layout(model, protocol["rank"])
    assert protocol["architecture"] == {
        "layers": len(groups),
        "hidden": groups[0]["hidden"],
        "intermediate": groups[0]["intermediate"],
    }
    names = [n for g in groups for n in (*g["incoming"], g["outgoing"])]
    original, replay = parents["original"], parents["replay"]
    data = json.loads((original / "inputs.json").read_text())
    assert sha256(original / "inputs.json") == protocol["inputs_sha256"]
    cases = {c["case"]: c for c in data["cases"]}
    references = {
        (f["case"], f["condition"]["id"]): f
        for f in json.loads((original / "fits.json").read_text())
    }
    tokens = json.loads((original / "tokenization.json").read_text())
    samples = {}
    device = next(model.parameters()).device
    for case in cases.values():
        for bank in ("train", "preserve", "target_val", *protocol["generation_banks"]):
            for row in case[bank]:
                sample = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
                assert tokens[row["id"]] == {
                    "ids": sample.input_ids[0].tolist(),
                    "prompt_length": sample.prompt_length,
                }
                samples[row["id"]] = sample
    write(output / "protocol.json", protocol)
    write(
        output / "identity.json",
        {"base_fingerprint": identity, "budgets": budgets, "layer_counts": counts},
    )
    backups = {n: model.get_submodule(n).weight.detach().cpu().clone() for n in names}
    discovery = {}
    records = []
    observations = 0
    base_controls = 0

    def restore(case):
        nonlocal base_controls
        with torch.no_grad():
            for n, value in backups.items():
                model.get_submodule(n).weight.copy_(value)
        model.eval().requires_grad_(False)
        assert fingerprint(model) == identity
        baseline = json.loads((original / f"case-{case['case']}-base.json").read_text())
        for bank in ("train", "utility_test"):
            row = case[bank][0]
            assert (
                generate(model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
                == baseline[bank][0]
            )
            base_controls += 1

    for request in protocol["requests"]:
        check_time()
        c, condition = request["case"], request["condition"]
        case = cases[c]
        label = condition["id"]
        prefix = f"case-{c}-{label}"
        reference = references[(c, label)]
        task = [samples[r["id"]] for r in case["train"]]
        preserved = [samples[r["id"]] for r in case["preserve"]]
        measured = [*task, *preserved, *[samples[r["id"]] for r in case["target_val"]]]
        if c not in discovery:
            restore(case)
            captured = []
            started = time.monotonic()
            for sample in task:
                check_time()
                factors, _ = capture_response_factors(
                    model, sample, names, loss_scale=protocol["loss_scale"]
                )
                captured.append(factors)
            scores = [
                projection_scores(captured, incoming=g["incoming"], outgoing=g["outgoing"])
                for g in groups
            ]
            discovery[c] = {"seconds": time.monotonic() - started, "backward_passes": len(task)}
            selected = supports(groups, counts, scores, family="paired_mean", seed=protocol["seed"])
            frozen = next(
                r
                for r in protocol["requests"]
                if r["case"] == c and r["condition"]["family"] == "paired_mean"
            )
            assert _serialized(selected) == frozen["supports"]
            write(output / f"case-{c}-discovery.json", discovery[c])
            del captured, factors, scores
        print("FIT_START", c, label, request["role"], flush=True)
        started = time.monotonic()
        if condition["family"] == "lora":
            fitted = attach_lora(model, names, rank=protocol["rank"], seed=condition["seed"])
            parameters = [p for p in fitted.parameters() if p.requires_grad]
            owner = None
        else:
            owner = PairedMLPPatch(model, _selections(request["supports"]))
            fitted = model
            parameters = list(owner.trainable_parameters())
        attach_seconds = time.monotonic() - started
        assert sum(p.numel() for p in parameters) == request["trainable_scalars"]

        # Bind this fit before later reloads reuse the local owner/model names.
        def save_patch(directory, step, patch=owner, adapted=fitted):
            if patch is None:
                adapted.save_pretrained(directory, safe_serialization=True)
            else:
                patch.save(directory, metadata={"case": c, "condition": condition, "step": step})

        observer = Observer(
            fitted,
            parameters,
            task,
            preserved,
            protocol,
            prefix,
            output,
            save_patch,
            check_time,
            diagnostic=request["role"] == "known_failure_diagnostic",
        )
        history = fit(
            fitted,
            parameters,
            task,
            preserved,
            protocol,
            condition,
            check_time,
            before_update=observer.before_update,
            after_update=observer.after_update,
        )
        assert history == reference["history"], (
            "Monitoring altered the reference training trajectory"
        )
        final_directory = output / f"{prefix}-full-adapter"
        save_started = time.monotonic()
        save_patch(final_directory, protocol["steps"])
        final_save_seconds = time.monotonic() - save_started
        tensor_name = "adapter_model.safetensors" if owner is None else "patch.safetensors"
        assert _same_tensors(
            load_file(str(final_directory / tensor_name)),
            load_file(str(original / f"{prefix}-adapter" / tensor_name)),
        )
        final_live = teacher(fitted, measured)
        assert _same_tensors(
            final_live, load_file(str(replay / f"{prefix}-live-teacher.safetensors"))
        )
        save_file(final_live, output / f"{prefix}-final-live-teacher.safetensors")
        record = {
            "case": c,
            "condition": condition,
            "role": request["role"],
            "history": history,
            "selected_step": observer.selected_step,
            "checkpoints": observer.snapshots,
            "training_seconds": observer.training_seconds,
            "monitoring_seconds": observer.monitoring_seconds,
            "diagnostic_seconds": observer.diagnostic_seconds,
            "attach_seconds": attach_seconds,
            "final_serialization_seconds": final_save_seconds,
            "discovery_seconds": discovery[c]["seconds"]
            if condition["family"] == "paired_mean"
            else 0,
            "final_parameters_exact": True,
            "final_history_exact": True,
            "final_live_teacher_exact": True,
            "trainable_scalars": sum(p.numel() for p in parameters),
            "states": {},
        }
        model = unload_lora(fitted) if owner is None else owner.unload()
        del fitted, owner, parameters, observer, final_live, save_patch
        assert fingerprint(model) == identity
        for state in ("early", "full") if record["selected_step"] is not None else ("full",):
            check_time()
            directory = output / f"{prefix}-{state}-adapter"
            if condition["family"] == "lora":
                fitted = load_lora(
                    model, directory, names, rank=protocol["rank"], seed=condition["seed"]
                )
                owner = None
            else:
                owner = PairedMLPPatch.load(model, directory)
                fitted = model
            live = teacher(fitted, measured)
            step = record["selected_step"] if state == "early" else protocol["steps"]
            checkpoint = load_file(str(output / f"{prefix}-step-{step}-calibration.safetensors"))
            n = len(checkpoint["labels"])
            assert torch.equal(live["logits"][:n], checkpoint["logits"])
            assert torch.equal(live["labels"][:n], checkpoint["labels"])
            if state == "full":
                assert _same_tensors(
                    live, load_file(str(replay / f"{prefix}-live-teacher.safetensors"))
                )
            save_file(live, output / f"{prefix}-{state}-live-teacher.safetensors")
            model = merge_lora(fitted) if owner is None else owner.merge_and_unload()
            del fitted, owner
            merged = teacher(model, measured)
            save_file(merged, output / f"{prefix}-{state}-merged-teacher.safetensors")
            metrics = teacher_metrics(merged, len(task), len(preserved))
            m = len(task) + len(preserved)
            count = int(merged["lengths"][:m].sum())
            merged_calib = {
                "logits": merged["logits"][:count],
                "labels": merged["labels"][:count],
                "lengths": merged["lengths"][:m],
            }
            generated = {}
            for bank in protocol["generation_banks"]:
                generated[bank] = []
                for row in case[bank]:
                    check_time()
                    generated[bank].append(
                        generate(
                            model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"]
                        )
                    )
                    observations += 1
            write(output / f"{prefix}-{state}.json", generated)
            if state == "full":
                assert _same_tensors(
                    merged, load_file(str(original / f"{prefix}-teacher.safetensors"))
                )
                old = json.loads((original / f"{prefix}.json").read_text())
                assert all(generated[b] == old[b] for b in protocol["generation_banks"])
            record["states"][state] = {
                "step": step,
                "metrics": metrics,
                "merged_calibration": calibration(
                    merged_calib, len(task), protocol["gate_probability"]
                ),
                "compact_reload_teacher_exact": True,
                "parent_merged_teacher_exact": state == "full",
                "parent_merged_generations_exact": state == "full",
                "nominal_prequantization_update_l2": nominal_update_norm(
                    directory, condition["family"]
                ),
            }
            restore(case)
            del live, merged, checkpoint, merged_calib
            gc.collect()
        record["base_restored"] = True
        records.append(record)
        write(output / "fits.json", records)
        print("FIT_RESTORED", c, label, "early_step", record["selected_step"], flush=True)
    assert len(records) == protocol["new_fits"]
    write(
        output / "completion.json",
        {
            "status": "complete",
            "new_fits": len(records),
            "observations": observations,
            "base_generation_controls": base_controls,
            "base_restored": True,
            "final_fingerprint": fingerprint(model),
        },
    )
    print("COMPLETE checkpoint-study", flush=True)
    return records


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    runtime = {n: importlib.metadata.version(n) for n in protocol["runtime"]}
    assert all(runtime[k].split("+")[0] == v for k, v in protocol["runtime"].items())
    parents = find_parents(Path("/kaggle/input"), protocol)
    print("STAGE parent_manifests_verified", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded checkpoint study exceeded runtime")

    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    execute(model, tokenizer, parents, protocol, args.output, check_time)


if __name__ == "__main__":
    main()
