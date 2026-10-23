"""Bounded exploratory boundary adaptation with frozen comparisons and data roles."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.paired_mlp.model import attach_lora, load_lora, merge_lora, unload_lora
from experiments.paired_mlp.training import fit
from experiments.relation_learning.gradient_agreement import shared_gradient_geometry
from experiments.relation_learning.update_prediction import compare, predict_loss, preview
from neuronpatch import NeuronPatch
from neuronpatch.losses import completion_nll
from neuronpatch.patch import _replace_submodule, fingerprint
from neuronpatch.supervision import chat_response

from .probe_selection import capture_prompt, fit_probe, select
from .row_gradients import capture_rows
from .run import write
from .transfer_data import teacher_supervision


def projection_names(model):
    grouped, lora = [], []
    for i in range(len(model.model.layers)):
        root = f"model.layers.{i}"
        grouped.extend(f"{root}.mlp.{p}" for p in ("gate_proj", "up_proj"))
        lora.extend(f"{root}.self_attn.{p}_proj" for p in ("q", "k", "v", "o"))
        lora.extend(f"{root}.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj"))
    return grouped, lora


def prompt_inputs(tokenizer, row, device):
    ids = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    ids = torch.tensor([ids], device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@torch.no_grad()
def generate(model, tokenizer, rows, spec, path, check_time):
    result = []
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    for row in rows:
        check_time()
        inputs = prompt_inputs(tokenizer, row, model.device)
        length = inputs["input_ids"].shape[1]
        if length + spec["max_new_tokens"] > spec["max_context_tokens"]:
            raise ValueError("Generation context cap exceeded")
        tokens = model.generate(
            **inputs,
            max_new_tokens=spec["max_new_tokens"],
            do_sample=False,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )[0, length:].tolist()
        if not tokens:
            raise ValueError("Empty token sequence")
        result.append(
            {
                "id": row["id"],
                "prompt_ids": inputs["input_ids"][0].tolist(),
                "tokens": tokens,
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "ended": tokens[-1] in eos,
            }
        )
        write(path, result)
        if len(result) % 25 == 0:
            print("GENERATED", path.name, len(result), flush=True)
    return result


def tokenize(tokenizer, bank, spec, device):
    return [
        chat_response(
            tokenizer,
            row["messages"],
            row["target"],
            max_length=spec["max_sequence_tokens"],
            device=device,
        )
        for row in bank
    ]


def loss_for(model, sample):
    logits = model(
        input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
    ).logits
    loss = completion_nll(logits, sample.input_ids, sample.completion_mask, sample.attention_mask)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite completion loss")
    return loss


@torch.no_grad()
def losses(model, examples):
    return [float(loss_for(model, sample)) for sample in examples]


def validation_score(values, records):
    banks = {kind: [] for kind in ("injected", "clean", "direct", "quoted")}
    for value, row in zip(values, records, strict=True):
        banks[row["id"].split(":")[-2]].append(value)
    if not all(banks.values()):
        raise ValueError("All four validation kinds required")
    means = {k: sum(v) / len(v) for k, v in banks.items()}
    return {"macro_nll": sum(means.values()) / 4, "kind_nll": means}


def detach_grouped(owner, *, merge=False):
    """Experiment-owned lifecycle; only mutate the exact currently attached layers."""
    with torch.no_grad():
        for name, layer in owner.layers.items():
            if owner.model.get_submodule(name) is not layer:
                raise ValueError("Grouped attachment ownership changed")
            if merge:
                selected = layer.base.weight.index_select(0, layer.indices).float()
                delta = layer.delta.index_select(0, layer.groups)
                layer.base.weight.index_copy_(
                    0, layer.indices, (selected + delta).to(layer.base.weight.dtype)
                )
            _replace_submodule(owner.model, name, layer.base)
    return owner.model.eval().requires_grad_(False)


def update_observers(model, examples, spec, path, check_time):
    pending, forecasts, measured = {}, [], []

    def before(step, row, optimizer):
        if step not in spec["diagnostics"]["observer_steps"]:
            return
        proposal = preview(
            optimizer, max_scalars=spec["diagnostics"]["optimizer_preview_max_scalars"]
        )
        predictions = {}
        for bank, samples in examples.items():
            pieces = []
            for sample in samples:
                check_time()
                pieces.append(predict_loss(proposal, loss_for(model, sample)))
            predictions[bank] = {
                key: sum(p[key] for p in pieces) / len(pieces) for key in pieces[0]
            }
        pending[step] = (proposal, predictions)
        forecasts.append({"step": step, "predictions": predictions})
        write(path.with_name(path.stem + "-forecasts.json"), forecasts)

    def after(step, row, optimizer):
        if step not in pending:
            return
        proposal, predictions = pending.pop(step)
        record = {"step": step, "banks": {}}
        for bank, samples in examples.items():
            check_time()
            values = losses(model, samples)
            record["banks"][bank] = compare(
                proposal,
                predictions[bank],
                torch.tensor(sum(values) / len(values), dtype=torch.float64),
            )
        measured.append(record)
        write(path, measured)

    return before, after


def discover(model, tokenizer, train_rows, task_examples, spec, names, output, check_time):
    profiles, labels, profile_ids = {name: [] for name in names}, [], []
    for row in train_rows:
        if row["kind"] != "clean" and not (
            row["kind"] == "injected" and row["family"] == "redirect"
        ):
            continue
        check_time()
        captured = capture_prompt(model, prompt_inputs(tokenizer, row, model.device), names)
        labels.append(int(row["kind"] == "injected"))
        profile_ids.append(row["id"])
        for name in names:
            profiles[name].append(captured[name])
    profiles = {name: torch.stack(rows) for name, rows in profiles.items()}
    save_file(profiles, output / "training-profiles.safetensors")
    write(output / "profile-metadata.json", {"ids": profile_ids, "labels": labels})
    selections = {f"{a['family']}-{a['seed']}": {} for a in spec["arms"] if a["family"] != "lora"}
    probes = {}
    for ordinal, name in enumerate(names):
        check_time()
        probe = fit_probe(profiles[name], torch.tensor(labels))
        probes[name] = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in probe.items()
        }
        for arm in spec["arms"]:
            if arm["family"] == "lora":
                continue
            key = f"{arm['family']}-{arm['seed']}"
            selections[key][name] = select(
                profiles[name],
                probe,
                count=spec["grouped"]["selected_rows_per_projection"],
                random_seed=None if arm["family"] == "probe_grouped" else arm["seed"] + ordinal,
            )
    write(output / "probes.json", probes)
    write(output / "supports.json", selections)
    del profiles, probes
    # One backward per example supplies all supports; only selected row derivatives are retained.
    union = {
        name: sorted({i for support in selections.values() for i in support[name]["indices"]})
        for name in names
    }
    geometry = []
    for example_index, sample in enumerate(task_examples[:2]):
        check_time()
        gradients = capture_rows(model, sample, union, loss_scale=spec["loss_scale"])
        for arm, support in selections.items():
            for name in names:
                positions = [union[name].index(i) for i in support[name]["indices"]]
                record = shared_gradient_geometry(
                    gradients[name][positions], support[name]["groups"]
                )
                geometry.append(
                    {
                        "example_index": example_index,
                        "arm": arm,
                        "projection": name,
                        "retained_energy_fraction": record["retained_energy_fraction"],
                        "group_sizes": record["group_sizes"],
                    }
                )
    write(output / "group-gradient-geometry.json", geometry)
    return selections


def execute(model, tokenizer, data, spec, output, check_time):
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", spec)
    original = fingerprint(model)
    names, lora_names = projection_names(model)
    banks, supervision = {}, {}
    for bank in ("train", "calibration"):
        rows = data["splits"][bank]
        teachers = generate(
            model,
            tokenizer,
            [r for r in rows if r["kind"] == "clean"],
            spec,
            output / f"teachers-{bank}.json",
            check_time,
        )
        supervision[bank] = teacher_supervision(rows, teachers)
        if len(supervision[bank]["task"]) < spec["minimum_teachers"][bank]:
            raise ValueError("Insufficient completed clean teachers")
        banks[bank] = {
            key: tokenize(tokenizer, supervision[bank][key], spec, model.device)
            for key in ("task", "preserve")
        }
    write(output / "supervision.json", supervision)
    generate(
        model,
        tokenizer,
        data["splits"]["confirmation"],
        spec,
        output / "base-confirmation.json",
        check_time,
    )
    started = time.monotonic()
    supports = discover(
        model,
        tokenizer,
        data["splits"]["train"],
        banks["train"]["task"],
        spec,
        names,
        output,
        check_time,
    )
    write(
        output / "discovery.json",
        {"seconds": time.monotonic() - started, "training_prompts_only": True},
    )
    calibration = banks["calibration"]["task"] + banks["calibration"]["preserve"]
    calibration_rows = supervision["calibration"]["task"] + supervision["calibration"]["preserve"]
    clean_positions = [
        i
        for i, r in enumerate(supervision["calibration"]["preserve"])
        if r["id"].endswith(":clean:none")
    ][:2]
    observed = {
        "task": banks["calibration"]["task"][:2],
        "clean": [banks["calibration"]["preserve"][i] for i in clean_positions],
    }
    fits = []

    def attach(arm, directory=None):
        if arm["family"] == "lora":
            args = dict(rank=spec["lora"]["rank"], seed=arm["seed"])
            fitted = (
                attach_lora(model, lora_names, **args)
                if directory is None
                else load_lora(model, directory, lora_names, **args)
            )
            return fitted, None
        support = supports[f"{arm['family']}-{arm['seed']}"]
        owner = (
            NeuronPatch(
                model,
                {n: s["indices"] for n, s in support.items()},
                axis="output",
                groups={n: s["groups"] for n, s in support.items()},
                delta_dtype=torch.float32,
            )
            if directory is None
            else NeuronPatch.load(model, directory)
        )
        return model, owner

    for arm in spec["arms"]:
        for rate in spec["learning_rates"]:
            check_time()
            label = f"{arm['family']}-{arm['seed']}-{rate}"
            print("FIT_START", label, flush=True)
            started = time.monotonic()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            fitted, owner = attach(arm)
            parameters = [p for p in fitted.parameters() if p.requires_grad]
            count = sum(p.numel() for p in parameters)
            expected = spec["expected_budgets"][
                "lora_rank1_all_projections"
                if arm["family"] == "lora"
                else "grouped_two_groups_gate_up"
            ]
            if count != expected:
                raise ValueError("Actual trainable budget differs from frozen protocol")
            before, after = update_observers(
                fitted, observed, spec, output / f"{label}-observer.json", check_time
            )
            history = fit(
                fitted,
                parameters,
                banks["train"]["task"],
                banks["train"]["preserve"],
                spec,
                {"id": label, "lr": rate},
                check_time,
                before_update=before,
                after_update=after,
            )
            validation = losses(fitted, calibration)
            live = losses(fitted, calibration[:2])
            live_parameters = {
                name: p.detach().cpu().clone()
                for name, p in fitted.named_parameters()
                if p.requires_grad
            }
            directory = output / f"{label}-adapter"
            if owner is None:
                fitted.save_pretrained(directory, safe_serialization=True)
                model = unload_lora(fitted)
            else:
                owner.save(directory, metadata={"arm": arm, "rate": rate})
                model = detach_grouped(owner)
            del fitted, owner, parameters, before, after
            model.eval().requires_grad_(False)
            if fingerprint(model) != original:
                raise ValueError("Base mutated by training/unload")
            fitted, owner = attach(arm, directory)
            reloaded_parameters = dict(fitted.named_parameters())
            if not all(
                torch.equal(value, reloaded_parameters[name].detach().cpu())
                for name, value in live_parameters.items()
            ):
                raise ValueError("Reload changes adapter tensors")
            if losses(fitted, calibration[:2]) != live:
                raise ValueError("Reload changes calibration losses")
            model = unload_lora(fitted) if owner is None else detach_grouped(owner)
            del fitted, owner, live_parameters, reloaded_parameters
            model.eval().requires_grad_(False)
            record = {
                "id": label,
                "arm": arm,
                "lr": rate,
                "trainable_scalars": count,
                "history": history,
                "validation": validation_score(validation, calibration_rows),
                "validation_losses": validation,
                "reload_probe_exact": True,
                "reload_tensors_exact": True,
                "base_restored": True,
                "seconds": time.monotonic() - started,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0,
                "patch_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            }
            fits.append(record)
            write(output / "fits.json", fits)
            gc.collect()
    choices = []
    for arm in spec["arms"]:
        candidates = [row for row in fits if row["arm"] == arm]
        choices.append(min(candidates, key=lambda r: (r["validation"]["macro_nll"], r["lr"]))["id"])
    write(output / "choices.json", choices)  # Frozen before any adapted confirmation generation.
    for label in choices:
        check_time()
        arm = next(r["arm"] for r in fits if r["id"] == label)
        changed = lora_names if arm["family"] == "lora" else names
        backups = {}
        for name in changed:
            weight = model.get_submodule(name).weight
            indices = (
                None
                if arm["family"] == "lora"
                else supports[f"{arm['family']}-{arm['seed']}"][name]["indices"]
            )
            selected = weight if indices is None else weight[indices]
            backups[name] = (indices, selected.detach().cpu().clone())
        fitted, owner = attach(arm, output / f"{label}-adapter")
        live = losses(fitted, calibration[:2])
        model = merge_lora(fitted) if owner is None else detach_grouped(owner, merge=True)
        del fitted, owner
        model.eval().requires_grad_(False)
        merged = losses(model, calibration[:2])
        write(output / f"{label}-merge.json", {"live_nll": live, "merged_nll": merged})
        generate(
            model,
            tokenizer,
            data["splits"]["confirmation"],
            spec,
            output / f"{label}-confirmation.json",
            check_time,
        )
        with torch.no_grad():
            for name, (indices, weight) in backups.items():
                parameter = model.get_submodule(name).weight
                if indices is None:
                    parameter.copy_(weight)
                else:
                    parameter.index_copy_(
                        0,
                        torch.tensor(indices, device=parameter.device),
                        weight.to(parameter.device),
                    )
        del backups
        if fingerprint(model) != original:
            raise ValueError("Merged evaluation base restoration failed")
        gc.collect()
    write(
        output / "completion.json",
        {
            "status": "complete",
            "fits": len(fits),
            "selected": choices,
            "base_restored": True,
            "original_fingerprint": original,
            "decision": "pending independent verification and semantic review",
        },
    )
    print("COMPLETE boundary-transfer", flush=True)
    return fits


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.data / "protocol.json").read_text())
    if {k: importlib.metadata.version(k).split("+")[0] for k in spec["runtime"]} != spec["runtime"]:
        raise ValueError("Runtime mismatch")
    raw = (args.data / "inputs.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["inputs_sha256"]:
        raise ValueError("Input hash mismatch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > spec["runtime_limit_seconds"]:
            raise TimeoutError("Bounded transfer runtime exceeded")

    torch.set_num_threads(2)
    torch.manual_seed(spec["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation=spec["attn_implementation"],
    )
    model.eval().requires_grad_(False)
    execute(model, tokenizer, json.loads(raw), spec, args.output, check_time)


if __name__ == "__main__":
    main()
