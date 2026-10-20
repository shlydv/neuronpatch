"""Bounded comparison of protected rows, matching row controls and LoRA."""

import argparse
import gc
import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch.layers import RowPatch
from neuronpatch.patch import fingerprint

from .helpers import Examples, write_json
from .prototype import InputSpan, ProtectedRowEdit, protected_span


def choose_anchors(rows, predictions, examples, protocol):
    by_id = {row["id"]: row for row in rows}
    eligible = []
    for prediction in predictions:
        row = by_id[prediction["id"]]
        length = len(examples.prefix(row)) + len(prediction["generated_token_ids"])
        if (
            row["kind"] == "answerable"
            and prediction["em"] == 1
            and not prediction.get("abstained", False)
            and 0 < len(prediction["generated_token_ids"])
            and length <= protocol["anchor_max_tokens"]
        ):
            eligible.append((length, row["id"], row, prediction))
    anchors, articles, total = [], set(), 0
    for length, _, row, prediction in sorted(eligible):
        if row["article"] in articles or total + length > protocol["anchor_token_budget"]:
            continue
        anchors.append({**row, "_target_ids": prediction["generated_token_ids"]})
        articles.add(row["article"])
        total += length
        if len(anchors) == protocol["anchor_count"]:
            break
    if len(anchors) < protocol["minimum_anchors"]:
        raise ValueError("Not enough baseline-correct calibration anchors under the fixed rule")
    return anchors


@torch.no_grad()
def capture_spans(model, targets, anchors, examples):
    values = {name: [] for name in targets}
    handles = []
    try:
        for name in targets:

            def capture(module, inputs, name=name):
                values[name].append(inputs[0][0].detach().cpu())

            handles.append(model.get_submodule(name).register_forward_pre_hook(capture))
        for row in anchors:
            inputs = examples.inputs(row, training=True)
            model(
                **{k: v for k, v in inputs.items() if k != "labels"},
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    spans, report = {}, {}
    for name in targets:
        started = time.monotonic()
        data = torch.cat(values.pop(name)).to(examples.device)
        span = protected_span(data)
        spans[name] = span
        report[name] = {
            "tokens": span.samples,
            "rank": span.rank,
            "free_input_directions": span.free_directions,
            "relative_tolerance": span.relative_tolerance,
            "max_input_reconstruction_error": span.max_residual,
            "capture_dtype": str(data.dtype),
            "basis_dtype": str(span.basis.dtype),
            "svd_driver": "gesvd" if data.is_cuda else "CPU default",
            "svd_seconds": time.monotonic() - started,
        }
        print(f"STAGE geometry={name} rank={span.rank} free={span.free_directions}", flush=True)
    return spans, report


def select_rows(model, targets, spans, calibration, examples, count, *, budget=lambda: None):
    totals = {
        name: torch.zeros_like(model.get_submodule(name).weight, dtype=torch.float32)
        for name in targets
    }
    try:
        for name in targets:
            model.get_submodule(name).weight.requires_grad_(True)
        for index, row in enumerate(calibration):
            budget()
            model.zero_grad(set_to_none=True)
            (examples.loss(model, row) * 128).backward()
            for name in targets:
                gradient = model.get_submodule(name).weight.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite or missing discovery gradient")
                totals[name].add_(gradient.float(), alpha=1 / (128 * len(calibration)))
            if (index + 1) % 8 == 0:
                print(f"STAGE discovery examples={index + 1}", flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
    selected, report = {}, {}
    with torch.no_grad():
        for name in targets:
            gradient = totals.pop(name).double()
            basis = spans[name].basis
            allowed = gradient - (gradient @ basis) @ basis.T
            norms = torch.linalg.vector_norm(allowed, dim=1)
            selected[name] = norms.topk(count).indices.sort().values.cpu().tolist()
            report[name] = {
                "indices": selected[name],
                "scores": norms.cpu().tolist(),
                "allowed_gradient_fraction": float(
                    allowed.norm() / gradient.norm().clamp_min(1e-30)
                ),
            }
    return selected, report


@torch.no_grad()
def restore_base(model, backups):
    for name, value in backups.items():
        module = model.get_submodule(name)
        if isinstance(module, RowPatch):
            module = module.base
            model.set_submodule(name, module)
        elif hasattr(module, "get_base_layer"):
            module = module.get_base_layer()
            model.set_submodule(name, module)
        if type(module) is not torch.nn.Linear:
            raise ValueError("Cannot restore an unexpected projection type")
        module.weight.copy_(value.to(module.weight.device))
    model.requires_grad_(False).eval()


def fit(
    model,
    config,
    indices,
    spans,
    all_targets,
    anchors,
    training,
    examples,
    out,
    protocol,
    budget,
):
    torch.manual_seed(protocol["training_seed"])
    model.eval().requires_grad_(False)
    metadata = {**config, "protocol": protocol}
    if config["family"] == "lora":
        trained = get_peft_model(
            model,
            LoraConfig(
                r=protocol["lora_rank"],
                lora_alpha=protocol["lora_alpha"],
                target_modules=all_targets,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ).eval()
        parameters = [p for p in trained.parameters() if p.requires_grad]
        effective = None
    else:
        if config["family"] == "unprotected":
            active_spans = {
                name: InputSpan(
                    torch.empty(
                        span.basis.shape[0], 0, dtype=torch.float64, device=span.basis.device
                    ),
                    0,
                    0,
                    0,
                )
                for name, span in spans.items()
            }
        else:
            active_spans = spans
        trained = ProtectedRowEdit(model, indices, active_spans).eval()
        parameters = list(trained.trainable_parameters())
        effective = trained.effective_dimensions
    allocated = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=0)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=examples.device.type == "cuda",
        init_scale=128,
        growth_interval=1000,
    )
    order = list(range(len(training)))
    random.Random(protocol["training_seed"]).shuffle(order)
    history = []
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "config.json",
        {
            **metadata,
            "indices": indices,
            "allocated_parameters": allocated,
            "effective_linear_dimension": effective,
        },
    )
    for step, index in enumerate(order):
        budget()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in (training[index], anchors[step % len(anchors)]):
            loss = examples.loss(trained, row)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss * 0.5).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, protocol["gradient_clip"])
        before_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "step": step + 1,
                "correction_ce": losses[0],
                "preservation_ce": losses[1],
                "gradient_norm": float(norm) if torch.isfinite(norm) else None,
                "loss_scale": before_scale,
                "optimizer_skipped": scaler.get_scale() < before_scale,
            }
        )
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError("Nonfinite trained parameters")
        if (step + 1) % 16 == 0:
            write_json(out / "history.json", history)
            print(f"STAGE fit={config['name']} step={step + 1} ce={losses[0]:.4f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    write_json(out / "history.json", history)
    training_seconds = time.monotonic() - started
    context = trained.applied() if isinstance(trained, ProtectedRowEdit) else nullcontext(trained)
    with context as live:
        examples.evaluate(live, anchors, out / "anchors_live.json", budget=budget)
    export_started = time.monotonic()
    if config["family"] == "lora":
        trained.save_pretrained(out / "adapter", safe_serialization=True)
        trained.merge_and_unload(safe_merge=True)
    else:
        patch = trained.export()
        patch.save(out / "patch", metadata=metadata)
        # Match the exported FP16 deployment form without a second full model.
        with torch.no_grad():
            for name, layer in patch.layers.items():
                base, delta, row_ids = layer.base, layer.delta.detach(), layer.indices
                model.set_submodule(name, base)
                base.weight.index_add_(0, row_ids, delta.to(base.weight.dtype))
    artifact_directory = out / ("adapter" if config["family"] == "lora" else "patch")
    return {
        "allocated_parameters": allocated,
        "effective_linear_dimension": effective,
        "training_seconds": training_seconds,
        "export_seconds": time.monotonic() - export_started,
        "optimizer_skips": sum(row["optimizer_skipped"] for row in history),
        "artifact_bytes": sum(
            path.stat().st_size for path in artifact_directory.rglob("*") if path.is_file()
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest = json.loads((args.data / "manifest.json").read_text())
    data = {}
    for name, wanted in manifest["sha256"].items():
        raw = (args.data / f"{name}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != wanted:
            raise ValueError(f"Data hash mismatch: {name}")
        data[name] = json.loads(raw)
    if (
        protocol["data_manifest_sha256"]
        != hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Protocol does not identify this dataset manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("The actual model pilot requires CUDA")
    torch.set_num_threads(4)
    started = time.monotonic()

    def budget():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Predeclared pilot runtime limit reached")

    torch.manual_seed(protocol["training_seed"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("STAGE model_load", flush=True)
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
    print("STAGE model_ready", flush=True)
    if model.config.hidden_size != protocol["hidden_size"]:
        raise ValueError("Unexpected architecture")
    targets = [f"model.layers.{layer}.mlp.up_proj" for layer in protocol["layers"]]
    all_targets = [
        f"model.layers.{layer}.mlp.up_proj" for layer in range(model.config.num_hidden_layers)
    ]
    examples = Examples(tokenizer, torch.device("cuda:0"))
    for split in ("train", "cal", "val"):
        for row in data[split]:
            if examples.inputs(row, training=True)["input_ids"].shape[1] > 512:
                raise ValueError("Pinned runtime tokenization exceeds the fixed length budget")
    import peft
    import transformers

    execution = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "data_manifest": manifest,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "fits": {},
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 32,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.pad_token_id,
            "model_generation_config": model.generation_config.to_dict(),
        },
    }
    write_json(args.output / "manifest.json", execution)
    base_hash = fingerprint(model)
    backups = {
        name: model.get_submodule(name).weight.detach().cpu().clone() for name in all_targets
    }
    baseline, baseline_seconds = {}, {}
    for name in ("cal", "val", "utility"):
        begin = time.monotonic()
        baseline[name] = examples.evaluate(
            model, data[name], args.output / f"base_{name}.json", budget=budget
        )
        baseline_seconds[name] = time.monotonic() - begin
    execution["baseline_seconds"] = baseline_seconds
    anchors = choose_anchors(data["cal"], baseline["cal"], examples, protocol)
    write_json(args.output / "anchors.json", anchors)
    begin = time.monotonic()
    spans, geometry = capture_spans(model, targets, anchors, examples)
    execution["protection_discovery_seconds"] = time.monotonic() - begin
    write_json(args.output / "geometry.json", geometry)
    begin = time.monotonic()
    selected, discovery = select_rows(
        model,
        targets,
        spans,
        data["cal"],
        examples,
        protocol["rows_per_layer"],
        budget=budget,
    )
    execution["row_discovery_seconds"] = time.monotonic() - begin
    write_json(args.output / "selection.json", discovery)
    generator = torch.Generator().manual_seed(protocol["random_seed"])
    random_rows = {
        name: torch.randperm(model.get_submodule(name).out_features, generator=generator)[
            : protocol["rows_per_layer"]
        ]
        .sort()
        .values.tolist()
        for name in targets
    }
    write_json(args.output / "random_rows.json", random_rows)
    if fingerprint(model) != base_hash:
        raise AssertionError("Discovery changed base weights")
    # Give every control the same label budget, including discovery questions.
    adaptation = data["train"] + data["cal"]
    if len(adaptation) != protocol["training_examples"] or protocol["epochs"] != 1:
        raise ValueError("This runner implements the frozen one-epoch adaptation pool")
    for config in protocol["fits"]:
        print(f"STAGE fit_start={config['name']}", flush=True)
        out = args.output / config["name"]
        torch.cuda.reset_peak_memory_stats()
        try:
            result = fit(
                model,
                config,
                random_rows if config["family"] == "random" else selected,
                spans,
                all_targets,
                anchors,
                adaptation,
                examples,
                out,
                protocol,
                budget,
            )
            for name, rows in (
                ("anchors_merged", anchors),
                ("val", data["val"]),
                ("utility", data["utility"]),
            ):
                examples.evaluate(model, rows, out / f"{name}.json", budget=budget)
            result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
            execution["fits"][config["name"]] = result
        finally:
            restore_base(model, backups)
            gc.collect()
            torch.cuda.empty_cache()
        if fingerprint(model) != base_hash:
            raise AssertionError("Base restoration hash mismatch")
        if examples.generate(model, data["val"][0])["text"] != baseline["val"][0]["text"]:
            raise AssertionError("Base generation changed after restoration")
        write_json(args.output / "manifest.json", execution)
        print(f"STAGE fit_complete={config['name']}", flush=True)
    execution["completed"] = True
    execution["runtime_seconds"] = time.monotonic() - started
    write_json(args.output / "manifest.json", execution)
    print("STAGE pilot_complete", flush=True)


if __name__ == "__main__":
    main()
