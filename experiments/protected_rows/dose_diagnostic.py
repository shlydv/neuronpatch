"""Separate trained-patch direction from refusal strength without retraining."""

import argparse
import gc
import hashlib
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from experiments.protected_rows.helpers import Examples, write_json
from neuronpatch import NeuronPatch
from neuronpatch.patch import fingerprint


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def exported_rows(model, directory):
    """Exercise the public exact-base loader, then detach its temporary wrappers."""
    loaded = NeuronPatch.load(model, directory)
    rows = {}
    for name, layer in loaded.layers.items():
        rows[name] = (layer.indices.detach().clone(), layer.delta.detach().clone())
        model.set_submodule(name, layer.base)
    return rows, loaded.base_fingerprint


@contextmanager
def merged_rows(model, backup, rows, scale):
    try:
        with torch.no_grad():
            for name, (indices, delta) in rows.items():
                weight = model.get_submodule(name).weight
                weight.copy_(backup[name])
                weight.index_add_(0, indices, (delta.float() * scale).to(weight.dtype))
        yield model
    finally:
        with torch.no_grad():
            for name, original in backup.items():
                weight = model.get_submodule(name).weight
                weight.copy_(original)
                if not torch.equal(weight, original):
                    raise AssertionError("Edited projection was not exactly restored")


class FirstTokenBias(LogitsProcessor):
    def __init__(self, prefix_length, token_id, bias):
        self.prefix_length, self.token_id, self.bias = prefix_length, token_id, bias

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] == self.prefix_length:
            scores[:, self.token_id] += self.bias
        return scores


class BiasView:
    def __init__(self, model, token_id, bias):
        self.model, self.token_id, self.bias = model, token_id, bias

    def generate(self, **kwargs):
        return self.model.generate(
            **kwargs,
            logits_processor=[
                FirstTokenBias(kwargs["input_ids"].shape[-1], self.token_id, self.bias)
            ],
        )


@torch.no_grad()
def signals(model, rows, examples, token_id, budget):
    result = []
    for row in rows:
        budget()
        logits = (
            model(**examples.inputs(row), use_cache=False, logits_to_keep=1).logits[0, -1].float()
        )
        unknown = logits[token_id].clone()
        logits[token_id] = -torch.inf
        odds = unknown - logits.logsumexp(0)
        if not torch.isfinite(odds):
            raise FloatingPointError("Nonfinite refusal log odds")
        result.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "unknown_log_odds": float(odds),
                "unknown_argmax_margin": float(unknown - logits.max()),
                "best_other_token": int(logits.argmax()),
            }
        )
    return result


def mean_shift(base, edited):
    assert [row["id"] for row in base] == [row["id"] for row in edited]
    return sum(b["unknown_log_odds"] - a["unknown_log_odds"] for a, b in zip(base, edited)) / len(
        base
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(args.protocol)
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != protocol["diagnostic_source_sha256"]
    ):
        raise ValueError("Diagnostic source differs from frozen protocol")
    execution = read(args.pilot / "manifest.json")
    if not execution.get("completed"):
        raise ValueError("The completed pilot is required")
    for name, expected in protocol["artifact_sha256"].items():
        if hashlib.sha256((args.pilot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    pilot_protocol = execution["protocol"]
    if (
        hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
        != pilot_protocol["data_manifest_sha256"]
    ):
        raise ValueError("Pilot data manifest differs")
    for name, expected in execution["data_manifest"]["sha256"].items():
        if hashlib.sha256((args.data / f"{name}.json").read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data bytes differ: {name}")
    started = time.monotonic()

    def budget():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Frozen diagnostic budget reached")

    torch.set_num_threads(4)
    torch.manual_seed(pilot_protocol["training_seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        pilot_protocol["model"], revision=pilot_protocol["revision"], local_files_only=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    unknown_ids = tokenizer.encode("UNKNOWN", add_special_tokens=False)
    if len(unknown_ids) != 1:
        raise ValueError("This frozen diagnostic requires a single UNKNOWN token")
    token_id = unknown_ids[0]
    print("STAGE diagnostic_model_load", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            pilot_protocol["model"],
            revision=pilot_protocol["revision"],
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    protected, base_hash = exported_rows(model, args.pilot / "protected-1/patch")
    unprotected, other_hash = exported_rows(model, args.pilot / "unprotected-1/patch")
    assert base_hash == other_hash and protected.keys() == unprotected.keys()
    assert all(torch.equal(protected[name][0], unprotected[name][0]) for name in protected)
    backup = {name: model.get_submodule(name).weight.detach().clone() for name in protected}
    examples = Examples(tokenizer, torch.device("cuda:0"))
    data = {name: read(args.data / f"{name}.json") for name in ("cal", "val", "utility")}
    original_base = {row["id"]: row for row in read(args.pilot / "base_val.json")}
    for row in data["val"][:4]:
        actual = examples.generate(BiasView(model, token_id, 0), row)
        assert actual["generated_token_ids"] == original_base[row["id"]]["generated_token_ids"]
    base_cal = signals(model, data["cal"], examples, token_id, budget)
    base_val = signals(model, data["val"], examples, token_id, budget)
    write_json(args.output / "base_cal_signals.json", base_cal)
    write_json(args.output / "base_val_signals.json", base_val)
    with merged_rows(model, backup, protected, 1):
        protected_cal = signals(model, data["cal"], examples, token_id, budget)
    target = mean_shift(base_cal, protected_cal)
    doses = []
    for scale in protocol["calibration_scale_grid"]:
        budget()
        with merged_rows(model, backup, unprotected, scale):
            values = (
                base_cal if scale == 0 else signals(model, data["cal"], examples, token_id, budget)
            )
        doses.append({"scale": scale, "shift": mean_shift(base_cal, values)})
        print(f"STAGE dose_calibration scale={scale} shift={doses[-1]['shift']:.5f}", flush=True)
    # One interpolation, chosen entirely on calibration; no validation-based tuning.
    for left, right in zip(doses, doses[1:]):
        if left["shift"] != right["shift"] and min(left["shift"], right["shift"]) < target < max(
            left["shift"], right["shift"]
        ):
            scale = left["scale"] + (target - left["shift"]) / (right["shift"] - left["shift"]) * (
                right["scale"] - left["scale"]
            )
            with merged_rows(model, backup, unprotected, scale):
                values = signals(model, data["cal"], examples, token_id, budget)
            doses.append(
                {"scale": scale, "shift": mean_shift(base_cal, values), "interpolated": True}
            )
            break
    chosen = min(doses, key=lambda row: (abs(row["shift"] - target), row["scale"]))
    record = {
        "protocol": protocol,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "zero_bias_base_probes_exact": 4,
        "unknown_token_id": token_id,
        "base_fingerprint": base_hash,
        "protected_calibration_shift": target,
        "calibration_doses": doses,
        "matched_scale": chosen["scale"],
        "matched_shift": chosen["shift"],
        "relative_matching_error": abs(chosen["shift"] - target) / max(abs(target), 1e-8),
        "reproduction": {},
    }
    write_json(args.output / "manifest.json", record)
    conditions = [
        ("protected_full", protected, 1, "protected-1"),
        ("unprotected_full", unprotected, 1, "unprotected-1"),
        ("unprotected_matched", unprotected, chosen["scale"], None),
        ("unknown_bias", None, target, None),
    ]
    for name, rows, scale, reference in conditions:
        print(f"STAGE dose_condition={name}", flush=True)
        context = (
            merged_rows(model, backup, rows, scale) if rows is not None else nullcontext(model)
        )
        with context:
            view = model if rows is not None else BiasView(model, token_id, scale)
            if rows is not None:
                values = signals(model, data["val"], examples, token_id, budget)
                write_json(args.output / name / "val_signals.json", values)
            for split in ("val", "utility"):
                predictions = examples.evaluate(
                    view, data[split], args.output / name / f"{split}.json", budget=budget
                )
                if reference:
                    expected = {
                        row["id"]: row for row in read(args.pilot / reference / f"{split}.json")
                    }
                    mismatches = [
                        row["id"]
                        for row in predictions
                        if row["generated_token_ids"] != expected[row["id"]]["generated_token_ids"]
                    ]
                    record["reproduction"][f"{name}/{split}"] = {
                        "n": len(predictions),
                        "mismatched_ids": mismatches,
                    }
                    write_json(args.output / "manifest.json", record)
                    if mismatches:
                        raise AssertionError(
                            "Cold-load exported patch generations differ from pilot"
                        )
        gc.collect()
    assert fingerprint(model) == base_hash
    record.update(completed=True, runtime_seconds=time.monotonic() - started)
    write_json(args.output / "manifest.json", record)
    print("STAGE dose_diagnostic_complete", flush=True)


if __name__ == "__main__":
    main()
