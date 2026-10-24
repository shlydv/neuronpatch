"""Independent LoRA arms with measured reference identity and verified exports.

The caller supplies frozen, tokenized data and an evaluation callback. Evaluation
runs only on the reloaded adapter, so reported behavior belongs to the artifact
that users can actually load. This module never selects checkpoints or examples.
"""

import json
import time
from pathlib import Path

import torch

from experiments.paired_mlp.model import attach_lora, load_lora, unload_lora
from neuronpatch.patch import fingerprint

from .preference_training import cache_references, fit_preferences, score


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_comparison(
    model,
    banks,
    preserved,
    *,
    names,
    rank,
    protocol,
    output,
    evaluate,
    check_time=lambda: None,
):
    """Run narrow/diverse × SFT/DPO from the same frozen weights.

    `evaluate(model, arm_directory)` must persist complete generated responses;
    its return value is saved verbatim. Failures leave partial artifacts but never
    a success manifest. Existing output directories are rejected to prevent mixed
    attempts. The returned model has no adapters and exactly its original weights.
    """
    if set(banks) != {"narrow", "diverse"} or any(not rows for rows in banks.values()):
        raise ValueError("Nonempty narrow and diverse banks required")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Frozen eval-mode base required")
    union = {}
    for rows in banks.values():
        if len({p.id for p in rows}) != len(rows):
            raise ValueError("Duplicate pair within bank")
        for pair in rows:
            if pair.id in union and union[pair.id].identity != pair.identity:
                raise ValueError("Pair ID changes identity across banks")
            union[pair.id] = pair
    if not set(p.id for p in banks["narrow"]).issubset(p.id for p in banks["diverse"]):
        raise ValueError("Diverse bank must include the narrow control")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = fingerprint(model)
    write(
        output / "protocol.json",
        {
            "training": protocol,
            "rank": rank,
            "modules": names,
            "base_identity": original,
            "banks": {
                k: [{"id": p.id, "identity": p.identity} for p in v] for k, v in banks.items()
            },
        },
    )
    started = time.monotonic()
    cache = cache_references(
        model, list(union.values()), base_identity=original, check_time=check_time
    )
    write(output / "references.json", cache)
    reference_seconds = time.monotonic() - started
    results = {}
    for coverage in ("narrow", "diverse"):
        for objective in ("sft", "dpo"):
            check_time()
            arm = f"{coverage}-{objective}"
            directory = output / arm
            directory.mkdir()
            fitted = attach_lora(model, names, rank=rank, seed=protocol["seed"])
            try:
                started = time.monotonic()
                history = fit_preferences(
                    fitted,
                    banks[coverage],
                    preserved,
                    cache,
                    base_identity=original,
                    objective=objective,
                    protocol=protocol,
                    check_time=check_time,
                )
                fit_seconds = time.monotonic() - started
                write(directory / "history.json", history)
                # Probe both alternatives, not just the response being reinforced.
                probes = (banks[coverage][0].chosen, banks[coverage][0].rejected)
                with torch.no_grad():
                    before = torch.stack([score(fitted, s) for s in probes]).cpu()
                parameter_count = sum(p.numel() for p in fitted.parameters() if p.requires_grad)
                fitted.save_pretrained(directory / "adapter", safe_serialization=True)
            finally:
                model = unload_lora(fitted)
            if fingerprint(model) != original:
                raise RuntimeError("Training mutated frozen base weights")
            reloaded = load_lora(
                model, directory / "adapter", names, rank=rank, seed=protocol["seed"]
            )
            try:
                with torch.no_grad():
                    after = torch.stack([score(reloaded, s) for s in probes]).cpu()
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                evaluation = evaluate(reloaded, directory)
            finally:
                model = unload_lora(reloaded)
            if fingerprint(model) != original:
                raise RuntimeError("Export/evaluation mutated frozen base weights")
            results[arm] = {
                "fit_seconds": fit_seconds,
                "trainable_parameters": parameter_count,
                "successful_steps": len(history),
                "forward_calls": sum(
                    r["forward_calls"]
                    + sum(retry["forward_calls"] for retry in r["gradient_retries"])
                    for r in history
                ),
                "evaluation": evaluation,
                "export_score_exact": True,
                "base_restored": True,
            }
            write(directory / "result.json", results[arm])
    write(
        output / "complete.json",
        {
            "base_identity": original,
            "reference_seconds": reference_seconds,
            "reference_forward_calls": 2 * len(union),
            "arms": results,
        },
    )
    return model, results
