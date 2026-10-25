"""Loss-term diagnostic with exact replay of a prior exported adapter.

The evaluation cases are exposed development data. No checkpoint is selected.
The prior artifact must be supplied locally or mounted as a Kaggle kernel input.
"""

import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.paired_mlp.model import load_lora, unload_lora
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .checkpointing import enable_decoder_recomputation
from .preference_experiment import run_comparison, write
from .preference_run import generate
from .preference_training import prepare_pair


def verify_prior(directory, identities):
    for relative, expected in identities.items():
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Prior artifact identity mismatch: " + relative)


def find_prior(root, identities):
    candidates = []
    for config in Path(root).rglob("diverse-sft/adapter/adapter_config.json"):
        directory = config.parent.parent
        try:
            verify_prior(directory, identities)
        except ValueError:
            continue
        candidates.append(directory)
    if len(candidates) != 1:
        raise ValueError(f"Expected one hash-matching prior adapter, found {len(candidates)}")
    return candidates[0]


def compare_replay(actual, expected, ids):
    indexed = {r["id"]: r for r in expected}
    if len(indexed) != len(expected) or [r["id"] for r in actual] != ids:
        raise ValueError("Replay membership differs")
    for row in actual:
        if row["id"] not in indexed:
            raise ValueError("Replay case absent from prior output")
        old = indexed[row["id"]]
        for key in ("prompt_ids", "tokens", "text", "ended"):
            if row[key] != old[key]:
                raise ValueError("Prior output replay differs: " + row["id"] + "/" + key)


def execute(model, tokenizer, data, spec, output, check_time, *, prior=None):
    if fingerprint(model) != spec["base_fingerprint"]:
        raise ValueError("Loaded base differs from frozen identity")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Frozen eval-mode base required")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", spec)
    prior = Path(prior) if prior is not None else find_prior("/kaggle/input", spec["prior_files"])
    verify_prior(prior, spec["prior_files"])
    by_id = {r["id"]: r for r in data["confirmation"]["cases"]}
    ids = spec["diagnostic_case_ids"]
    if len(by_id) != len(data["confirmation"]["cases"]) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate diagnostic case IDs")
    cases = [by_id[i] for i in ids]
    clean = [r for r in cases if r["kind"] == "clean"]
    if not clean or not cases:
        raise ValueError("Clean and final diagnostic cases required")
    if any(s < 1 or s >= spec["training"]["steps"] for s in spec["diagnostic_steps"]):
        raise ValueError("Diagnostic steps must precede the final checkpoint")
    membership = [
        {t["document_id"] for t in data[role]["tasks"]}
        for role in ("train", "calibration", "confirmation")
    ]
    if any(membership[i] & membership[j] for i in range(3) for j in range(i)):
        raise ValueError("Document leakage across roles")
    names = []
    for i in range(len(model.model.layers)):
        names.extend(f"model.layers.{i}.self_attn.{p}_proj" for p in ("q", "k", "v", "o"))
        names.extend(f"model.layers.{i}.mlp.{p}_proj" for p in ("gate", "up", "down"))
    enable_decoder_recomputation(model)
    replay_dir = output / "combined-replay"
    replay_dir.mkdir()
    reloaded = load_lora(
        model, prior / "adapter", names, rank=spec["rank"], seed=spec["training"]["seed"]
    )
    try:
        replay = generate(
            reloaded,
            tokenizer,
            cases,
            spec["generation"],
            replay_dir / "confirmation.json",
            check_time,
        )
    finally:
        model = unload_lora(reloaded)
    compare_replay(
        json.loads((replay_dir / "confirmation.json").read_text()),
        json.loads((prior / "confirmation.json").read_text()),
        ids,
    )
    if fingerprint(model) != spec["base_fingerprint"]:
        raise RuntimeError("Replay mutated frozen base")
    write(replay_dir / "result.json", {**replay, "exact_prior_output_replay": True})

    banks = {
        k: [
            prepare_pair(tokenizer, r, max_length=spec["max_sequence_tokens"], device=model.device)
            for r in data["train"][k + "_pairs"]
        ]
        for k in ("narrow", "diverse")
    }
    preserved = [
        chat_response(
            tokenizer,
            r["messages"],
            r["target"],
            max_length=spec["max_sequence_tokens"],
            device=model.device,
        )
        for r in data["train"]["clean"]
    ]
    diagnostic_seconds = {}

    def on_step(fitted, directory, row):
        if row["step"] not in spec["diagnostic_steps"]:
            return
        started = time.monotonic()
        # Preserve RNG state even if a future generation implementation consumes it.
        devices = [fitted.device.index] if fitted.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            summary = generate(
                fitted,
                tokenizer,
                clean,
                spec["generation"],
                directory / f"clean-step-{row['step']}.json",
                check_time,
            )
        elapsed = time.monotonic() - started
        diagnostic_seconds[directory.name] = diagnostic_seconds.get(directory.name, 0.0) + elapsed
        write(directory / f"clean-step-{row['step']}-result.json", {**summary, "seconds": elapsed})

    def evaluate(fitted, directory):
        started = time.monotonic()
        summary = generate(
            fitted,
            tokenizer,
            cases,
            spec["generation"],
            directory / "confirmation.json",
            check_time,
        )
        return {
            **summary,
            "evaluation_seconds": time.monotonic() - started,
            "intermediate_diagnostic_seconds": diagnostic_seconds.get(directory.name, 0.0),
        }

    model, results = run_comparison(
        model,
        banks,
        preserved,
        names=names,
        rank=spec["rank"],
        protocol=spec["training"],
        output=output / "comparison",
        evaluate=evaluate,
        check_time=check_time,
        arms=[("diverse", "sft", mode) for mode in ("task_only", "preservation_only")],
        on_step=on_step,
    )
    write(
        output / "complete.json",
        {
            "fits": len(results),
            "exact_prior_output_replay": True,
            "base_restored": fingerprint(model) == spec["base_fingerprint"],
            "timing_note": "fit_seconds includes separately reported intermediate generation time",
            "scope": "Exposed development cases; loss-term diagnosis, not fresh confirmation",
        },
    )


if __name__ == "__main__":
    from .preference_run import main

    main(execute_fn=execute)
