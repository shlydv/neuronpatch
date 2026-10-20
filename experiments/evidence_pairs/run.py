"""Bounded, budget-matched conditional-learning diagnostic; not a public recipe."""

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from experiments.protected_rows.helpers import Examples, write_json
from experiments.real_qa.response_loss import response_ce, response_logits
from neuronpatch import NeuronPatch
from neuronpatch.losses import preservation_kl
from neuronpatch.patch import fingerprint

from .objective import evidence_scores, separation_loss


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FP32Rows:
    """Scoped FP32 optimizer state/deltas with FP16 base; export through NeuronPatch.

    This intentionally stays experimental: the public row module currently follows
    the base dtype. The hook casts its small input/delta branch explicitly. Main
    evaluation uses the exported, FP16-merged weights, not this training hook.
    """

    def __init__(self, model, targets):
        self.model, self.targets, self.deltas, self.handles = model, targets, {}, []
        for name, indices in targets.items():
            module = model.get_submodule(name)
            assert type(module) is nn.Linear
            delta = nn.Parameter(
                torch.zeros(
                    len(indices),
                    module.in_features,
                    device=module.weight.device,
                    dtype=torch.float32,
                )
            )
            index = torch.tensor(indices, device=module.weight.device, dtype=torch.long)
            self.deltas[name] = delta

            def add_rows(module, inputs, output, delta=delta, index=index):
                update = F.linear(inputs[0].float(), delta).to(output.dtype)
                return output.index_add(-1, index, update)

            self.handles.append(module.register_forward_hook(add_rows))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def batch(examples, rows, *, training):
    """Left pad complete examples; labels remain contiguous supervised suffixes."""
    inputs = [examples.inputs(row, training=training) for row in rows]
    width = max(value["input_ids"].shape[1] for value in inputs)
    result = {}
    for key in inputs[0]:
        fill = (
            -100
            if key == "labels"
            else (examples.tokenizer.pad_token_id if key == "input_ids" else 0)
        )
        result[key] = torch.cat(
            [F.pad(value[key], (width - value[key].shape[1], 0), value=fill) for value in inputs]
        )
    # Explicit position IDs make left-padding behavior independent of HF defaults.
    result["position_ids"] = result["attention_mask"].cumsum(-1).sub(1).clamp_min(0)
    return result


def primary_losses(model, pair, examples, unknown):
    inputs = batch(examples, pair["primary"], training=True)
    logits, labels = response_logits(model, inputs)
    counts = (labels != -100).sum(-1)
    first = logits[torch.arange(2, device=logits.device), logits.shape[1] - counts]
    scores = evidence_scores(
        first[None],
        torch.tensor([pair["answer_token"]], device=logits.device),
        unknown_token=unknown,
    )
    # Equal weight to the two decisions, regardless of answer length.
    ce = torch.stack([response_ce(logits[i : i + 1], labels[i : i + 1]) for i in range(2)]).mean()
    return ce, scores


class FirstTokenControl(LogitsProcessor):
    def __init__(self, length, token, bias=0.0, force=False):
        self.length, self.token, self.bias, self.force = length, token, bias, force

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] == self.length:
            if self.force:
                scores.fill_(-torch.inf)
                scores[:, self.token] = 0
            else:
                scores[:, self.token] += self.bias
        return scores


class ControlView:
    def __init__(self, model, token, *, bias=0.0, force=False):
        self.model, self.token, self.bias, self.force = model, token, bias, force

    def generate(self, **kwargs):
        return self.model.generate(
            **kwargs,
            logits_processor=[
                FirstTokenControl(kwargs["input_ids"].shape[-1], self.token, self.bias, self.force)
            ],
        )


@torch.no_grad()
def signals(model, rows, examples, unknown, budget):
    values = []
    for row in rows:
        budget()
        logits = model(**examples.inputs(row), use_cache=False, logits_to_keep=1).logits[0, -1]
        logits = logits.float()
        u = logits[unknown].clone()
        a = examples.tokenizer.encode(row["answer"], add_special_tokens=False)[0]
        answer_score = float(u - logits[a]) if row["kind"] == "answerable" else None
        argmax = int(logits.argmax())
        logits[unknown] = -torch.inf
        values.append(
            {
                "id": row["id"],
                "argmax": argmax,
                "unknown_margin": float(u - logits.max()),
                "unknown_log_odds": float(u - logits.logsumexp(0)),
                "unknown_minus_answer": answer_score,
            }
        )
    return values


def generate(model, rows, examples, path, budget):
    result = []
    for i, row in enumerate(rows):
        budget()
        value = examples.generate(model, row)
        value.update({k: row.get(k) for k in ("pair_id", "role", "context_variant")})
        result.append(value)
        if (i + 1) % 64 == 0:
            print(f"EVAL {path.parent.name}/{path.stem} {i + 1}/{len(rows)}", flush=True)
    write_json(path, result)
    return result


def primary_rows(pairs):
    return [row for pair in pairs for row in pair["primary"]]


def all_rows(pairs):
    return [row for pair in pairs for row in pair["primary"] + pair["invariant"]]


def metrics(values):
    groups = {
        "supported": [
            v for v in values if v.get("role") == "primary" and v["kind"] == "answerable"
        ],
        "missing": [v for v in values if v.get("role") == "primary" and v["kind"] == "missing"],
        "invariant": [v for v in values if v.get("role") == "invariant"],
    }
    return {
        name: {
            "correct": sum(v["em"] for v in rows),
            "n": len(rows),
            "abstentions": sum(v["abstained"] for v in rows),
        }
        for name, rows in groups.items()
    }


def retention(values, reference, *, roles):
    now = {v["id"]: v for v in values}
    correct = [v for v in reference if v.get("role") in roles and v["em"] == 1]
    return sum(now[v["id"]]["em"] for v in correct) / len(correct) if correct else 1.0


def calibrate(model, rows, raw, signal_values, examples, unknown, protocol, out, base, budget):
    """Exact greedy first-token branch cache, then verify the chosen bias directly.

    A nonnegative one-position bias has only two possible greedy continuations:
    the original first token or UNKNOWN. We actually generate the latter; it is
    not assumed to terminate or obey the instruction. Check every raw argmax and
    directly regenerate the selected setting. Any mismatch aborts the experiment.
    Utility never enters this calibration and the bias is task-scoped.
    """
    forced = []
    max_bias = max(protocol["bias_grid"])
    for row, pred, sig in zip(rows, raw, signal_values):
        assert pred["id"] == sig["id"] == row["id"]
        assert pred["generated_token_ids"][0] == sig["argmax"], "Prefill/generation mismatch"
        if sig["argmax"] == unknown or sig["unknown_margin"] + max_bias < 0:
            forced.append(pred)
        else:
            budget()
            forced.append(
                {**pred, **examples.generate(ControlView(model, unknown, force=True), row)}
            )
    write_json(out / "cal_forced_unknown.json", forced)
    choices = []
    for bias in protocol["bias_grid"]:
        # Strict > handles equality conservatively; ties are verified below.
        predictions = [
            f if s["unknown_margin"] + bias > 0 else p
            for p, f, s in zip(raw, forced, signal_values)
        ]
        scored = metrics(predictions)
        preserved = retention(predictions, base, roles={"invariant"})
        supported_base = [p for p in base if p["role"] == "primary" and p["kind"] == "answerable"]
        supported_retention = retention(predictions, supported_base, roles={"primary"})
        balanced = sum(scored[k]["correct"] / scored[k]["n"] for k in ("supported", "missing")) / 2
        choices.append(
            {
                "bias": bias,
                "balanced_em": balanced,
                "supported_retention": supported_retention,
                "invariant_retention": preserved,
                "metrics": scored,
            }
        )
    eligible = [
        c for c in choices if min(c["supported_retention"], c["invariant_retention"]) >= 0.95
    ]
    # If no setting preserves the base, choose the least damaging measured option
    # for reporting, while explicitly marking it ineligible for promotion.
    best = max(
        eligible or choices,
        key=lambda c: (
            c["balanced_em"]
            if eligible
            else min(c["supported_retention"], c["invariant_retention"]),
            -c["bias"],
        ),
    )
    predicted = [
        f if s["unknown_margin"] + best["bias"] > 0 else p
        for p, f, s in zip(raw, forced, signal_values)
    ]
    direct = generate(
        ControlView(model, unknown, bias=best["bias"]),
        rows,
        examples,
        out / "cal_calibrated.json",
        budget,
    )
    assert all(
        a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(predicted, direct)
    ), "Branch-cache calibration differs from actual generation"
    result = {
        "chosen": best,
        "eligible": bool(eligible),
        "grid": choices,
        "branch_cache_verified_all_tokens": True,
    }
    write_json(out / "calibration.json", result)
    return result


def select(model, targets, pairs, examples, unknown, protocol, budget):
    totals = {
        name: torch.zeros(model.get_submodule(name).out_features, device=examples.device)
        for name in targets
    }
    try:
        for name in targets:
            model.get_submodule(name).weight.requires_grad_(True)
        for i, pair in enumerate(pairs):
            budget()
            model.zero_grad(set_to_none=True)
            _, scores = primary_losses(model, pair, examples, unknown)
            loss = separation_loss(scores, scores.detach())
            (loss * 128).backward()
            for name in targets:
                grad = model.get_submodule(name).weight.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise FloatingPointError("Invalid selection gradient")
                totals[name].add_((grad.float() / 128).square().sum(-1) / len(pairs))
            if (i + 1) % 16 == 0:
                print(f"SELECT train pairs={i + 1}", flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
    selected, random_rows, report = {}, {}, {}
    for i, name in enumerate(targets):
        selected[name] = (
            totals[name].topk(protocol["rows_per_layer"]).indices.sort().values.tolist()
        )
        generator = torch.Generator().manual_seed(protocol["random_seed"] + i)
        random_rows[name] = (
            torch.randperm(len(totals[name]), generator=generator)[: protocol["rows_per_layer"]]
            .sort()
            .values.tolist()
        )
        report[name] = {
            "selected": selected[name],
            "random": random_rows[name],
            "mean_squared_row_gradient": totals[name].cpu().tolist(),
        }
    return selected, random_rows, report


@torch.no_grad()
def references(model, pairs, examples, unknown, budget):
    result = {}
    for i, pair in enumerate(pairs):
        budget()
        _, scores = primary_losses(model, pair, examples, unknown)
        controls = {}
        for row in pair["invariant"]:
            logits, labels = response_logits(model, examples.inputs(row, training=True))
            controls[row["id"]] = logits.cpu()
        result[pair["pair_id"]] = {"scores": scores.cpu(), "controls": controls}
        if (i + 1) % 16 == 0:
            print(f"REFERENCE train pairs={i + 1}", flush=True)
    return result


@torch.no_grad()
def restore(model, backup):
    for name, original in backup.items():
        module = model.get_submodule(name)
        assert type(module) is nn.Linear
        module.weight.copy_(original.to(module.weight.device))
        assert torch.equal(module.weight, original.to(module.weight.device))
    model.requires_grad_(False).eval()


def fit(model, config, supports, pairs, refs, examples, unknown, protocol, out, budget):
    torch.manual_seed(protocol["training_seed"])
    model.eval().requires_grad_(False)
    editor = None
    if config["family"] == "lora":
        trained = get_peft_model(
            model,
            LoraConfig(
                r=protocol["lora_rank"],
                lora_alpha=protocol["lora_alpha"],
                target_modules=list(supports),
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ).eval()
        parameters = [p for p in trained.parameters() if p.requires_grad]
    else:
        trained = model
        editor = FP32Rows(model, supports)
        parameters = list(editor.deltas.values())
    assert all(p.dtype == torch.float32 for p in parameters)
    count = sum(p.numel() for p in parameters)
    assert count == protocol["trainable_parameters"]
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "config.json",
        {
            **config,
            "allocated_parameters": count,
            "supports": supports if editor else None,
            "training_dtype": "float32",
            "base_dtype": "float16",
        },
    )
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=0)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=examples.device.type == "cuda", init_scale=128, growth_interval=1000
    )
    order = []
    for epoch in range(protocol["epochs"]):
        indices = list(range(len(pairs)))
        random.Random(protocol["training_seed"] + epoch).shuffle(indices)
        order += [(epoch, index) for index in indices]
    history, started = [], time.monotonic()
    try:
        for step, (epoch, index) in enumerate(order):
            budget()
            pair = pairs[index]
            optimizer.zero_grad(set_to_none=True)
            ce, scores = primary_losses(trained, pair, examples, unknown)
            contrast = separation_loss(scores, refs[pair["pair_id"]]["scores"].to(scores.device))
            objective = protocol["primary_ce_weight"] * ce
            if config["objective"] == "ce+pair":
                objective = objective + protocol["pair_weight"] * contrast
            scaler.scale(objective).backward()
            # Each pair's two retained-fact contexts appear once across two epochs.
            control = pair["invariant"][epoch % 2]
            logits, labels = response_logits(trained, examples.inputs(control, training=True))
            ref = refs[pair["pair_id"]]["controls"][control["id"]].to(logits.device)
            keep_ce = response_ce(logits, labels)
            kl = preservation_kl(logits, ref, labels != -100, token_reduction="max")
            preservation = (
                protocol["control_ce_weight"] * keep_ce + protocol["control_kl_weight"] * kl
            )
            scaler.scale(preservation).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, protocol["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite training gradient; no silent skipped fits")
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                raise FloatingPointError("Gradient-scaler skip; experiment aborted")
            history.append(
                {
                    "step": step + 1,
                    "pair_id": pair["pair_id"],
                    "primary_ce": float(ce.detach()),
                    "pair_loss": float(contrast.detach()),
                    "control_ce": float(keep_ce.detach()),
                    "control_max_kl": float(kl.detach()),
                    "gradient_norm": float(norm),
                }
            )
            if (step + 1) % 32 == 0:
                print(
                    f"FIT {config['name']} step={step + 1}/{len(order)} "
                    f"ce={float(ce.detach()):.3f} pair={float(contrast.detach()):.3f}",
                    flush=True,
                )
        write_json(out / "history.json", history)
        if editor:
            editor.close()
            patch = NeuronPatch(model, supports, axis="output")
            with torch.no_grad():
                for name, layer in patch.layers.items():
                    layer.delta.copy_(editor.deltas[name].to(layer.delta.dtype))
            patch.save(
                out / "patch", metadata={**config, "precision": "FP32 training, FP16 export"}
            )
            with torch.no_grad():
                for name, layer in patch.layers.items():
                    model.set_submodule(name, layer.base)
                    layer.base.weight.index_add_(0, layer.indices, layer.delta)
        else:
            trained.save_pretrained(out / "adapter", safe_serialization=True)
            trained.merge_and_unload(safe_merge=True)
        write_json(
            out / "fit_summary.json",
            {
                "steps": len(history),
                "training_and_export_seconds": time.monotonic() - started,
                "trainable_parameters": count,
                "optimizer_skips": 0,
                "evaluation": (
                    "merged FP16 weights; artifact factors may use different storage dtypes"
                ),
            },
        )
    finally:
        if editor:
            editor.close()
    del optimizer, parameters, trained, editor
    gc.collect()
    if examples.device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    for name, expected in protocol["data_sha256"].items():
        assert digest(args.data / name) == expected, f"Data mismatch: {name}"
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    pairs = {k: read(args.data / f"{k}_pairs.json") for k in ("train", "cal", "val")}
    utility = read(args.data / "hellaswag.json")
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    unknown_ids = tokenizer.encode("UNKNOWN", add_special_tokens=False)
    assert len(unknown_ids) == 1
    unknown = unknown_ids[0]
    torch.manual_seed(protocol["training_seed"])
    started = time.monotonic()

    def budget():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Frozen experiment runtime limit exceeded")

    print("LOAD pinned Qwen3 4B FP16", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    assert model.config.hidden_size == 2560 and model.config.intermediate_size == 9728
    examples = Examples(tokenizer, torch.device("cuda:0"), generation_tokens=32)
    base_hash = fingerprint(model)
    targets = [f"model.layers.{i}.mlp.up_proj" for i in protocol["layers"]]
    backup = {name: model.get_submodule(name).weight.detach().cpu().clone() for name in targets}
    write_json(
        args.output / "base_identity.json",
        {
            "fingerprint": base_hash,
            "model": protocol["model"],
            "revision": protocol["revision"],
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
    )
    base_out = args.output / "base"
    base_out.mkdir()
    cal_rows = all_rows(pairs["cal"])
    base_cal = generate(model, cal_rows, examples, base_out / "cal_raw.json", budget)
    cal_signals = signals(model, cal_rows, examples, unknown, budget)
    write_json(base_out / "cal_signals.json", cal_signals)
    calibration = calibrate(
        model,
        cal_rows,
        base_cal,
        cal_signals,
        examples,
        unknown,
        protocol,
        base_out,
        base_cal,
        budget,
    )
    train_base = generate(
        model, primary_rows(pairs["train"]), examples, base_out / "train_primary.json", budget
    )
    train_errors = sum(not p["em"] for p in train_base if p["kind"] == "missing")
    cal_errors = (
        calibration["chosen"]["metrics"]["missing"]["n"]
        - calibration["chosen"]["metrics"]["missing"]["correct"]
    )
    proceed = train_errors >= protocol["headroom"]["train_missing_errors"] and (
        cal_errors >= protocol["headroom"]["calibrated_cal_missing_errors"]
    )
    write_json(
        args.output / "headroom.json",
        {
            "proceed": proceed,
            "train_missing_errors": train_errors,
            "calibrated_cal_missing_errors": cal_errors,
            "decision_uses": (
                "train and calibration only; validation remains unread until after gate"
            ),
        },
    )
    if not proceed:
        write_json(
            args.output / "completion.json",
            {
                "status": "headroom gate stopped training",
                "seconds": time.monotonic() - started,
                "base_restored": fingerprint(model) == base_hash,
            },
        )
        print("STOP no sufficient baseline headroom under frozen rule", flush=True)
        return
    # Validation is first opened for predictions after the train/cal-only gate.
    val_rows = all_rows(pairs["val"])
    generate(model, val_rows, examples, base_out / "val_raw.json", budget)
    write_json(base_out / "val_signals.json", signals(model, val_rows, examples, unknown, budget))
    generate(
        ControlView(model, unknown, bias=calibration["chosen"]["bias"]),
        val_rows,
        examples,
        base_out / "val_calibrated.json",
        budget,
    )
    generate(model, utility, examples, base_out / "utility.json", budget)
    refs = references(model, pairs["train"], examples, unknown, budget)
    selected, random_rows, selection = select(
        model, targets, pairs["train"], examples, unknown, protocol, budget
    )
    write_json(args.output / "selection.json", selection)
    for config in protocol["fits"]:
        budget()
        out = args.output / config["name"]
        supports = random_rows if config["family"] == "random" else selected
        try:
            fit(
                model,
                config,
                supports,
                pairs["train"],
                refs,
                examples,
                unknown,
                protocol,
                out,
                budget,
            )
            raw = generate(model, cal_rows, examples, out / "cal_raw.json", budget)
            sig = signals(model, cal_rows, examples, unknown, budget)
            write_json(out / "cal_signals.json", sig)
            calibrated = calibrate(
                model, cal_rows, raw, sig, examples, unknown, protocol, out, base_cal, budget
            )
            generate(model, val_rows, examples, out / "val_raw.json", budget)
            generate(
                ControlView(model, unknown, bias=calibrated["chosen"]["bias"]),
                val_rows,
                examples,
                out / "val_calibrated.json",
                budget,
            )
            write_json(
                out / "val_signals.json", signals(model, val_rows, examples, unknown, budget)
            )
            generate(model, utility, examples, out / "utility.json", budget)
        finally:
            restore(model, backup)
        print(f"DONE {config['name']} exact target restoration", flush=True)
    final_hash = fingerprint(model)
    assert final_hash == base_hash
    write_json(
        args.output / "completion.json",
        {
            "status": "all frozen fits complete",
            "fits": len(protocol["fits"]),
            "seconds": time.monotonic() - started,
            "base_restored": True,
            "final_fingerprint": final_hash,
        },
    )
    print("COMPLETE all comparisons, full base fingerprint restored", flush=True)


if __name__ == "__main__":
    main()
