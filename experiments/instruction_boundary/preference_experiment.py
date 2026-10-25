"""Independent LoRA arms with measured reference identity and verified exports.

The caller supplies frozen, tokenized data and an evaluation callback. Evaluation
runs only on the reloaded adapter, so reported behavior belongs to the artifact
that users can actually load. This module never selects checkpoints or examples.
"""

import hashlib
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
    arms=None,
    on_step=None,
    preservation_objective="nll",
    max_cache_bytes=None,
):
    """Run narrow/diverse × SFT/DPO from the same frozen weights.

    `evaluate(model, arm_directory)` must persist complete generated responses;
    its return value is saved verbatim. Failures leave partial artifacts but never
    a success manifest. Existing output directories are rejected to prevent mixed
    attempts. The returned model has no adapters and exactly its original weights.
    """
    if preservation_objective not in ("nll", "kl"):
        raise ValueError("Unknown preservation objective")
    if arms is None:
        arms = [
            (coverage, objective, "combined")
            for coverage in ("narrow", "diverse")
            for objective in ("sft", "dpo")
        ]
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("Nonempty unique arms required")
    for coverage, objective, terms in arms:
        if (
            coverage not in ("narrow", "diverse")
            or objective not in ("sft", "dpo")
            or terms not in ("combined", "task_only", "preservation_only")
        ):
            raise ValueError("Unknown comparison arm")
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
            "arms": arms,
            "preservation_objective": preservation_objective,
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
    retained = None
    retention_receipt = None
    if preservation_objective == "kl":
        from safetensors.torch import save_file

        from .response_retention import cache_responses, retention_loss

        started = time.monotonic()
        retained = cache_responses(
            model,
            preserved,
            base_identity=original,
            max_cache_bytes=max_cache_bytes,
            check_time=check_time,
        )
        path = output / "retention-reference.safetensors"
        save_file({str(i): r.reference_logits for i, r in enumerate(retained)}, path)
        retention_receipt = {
            "base_identity": original,
            "sample_sha256": [r.sample_sha256 for r in retained],
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tensor_bytes": sum(
                r.reference_logits.numel() * r.reference_logits.element_size() for r in retained
            ),
            "forward_calls": len(retained),
            "seconds": time.monotonic() - started,
        }
        write(output / "retention-reference.json", retention_receipt)
    results = {}
    for coverage, objective, loss_terms in arms:
        check_time()
        arm = f"{coverage}-{objective}"
        if loss_terms != "combined":
            arm += f"-{loss_terms}"
        if preservation_objective == "kl":
            arm += "-kl"
        directory = output / arm
        directory.mkdir()
        fitted = attach_lora(model, names, rank=rank, seed=protocol["seed"])
        try:
            initial_kl = None
            if retained is not None:
                with torch.no_grad():
                    initial_kl = [
                        float(retention_loss(fitted, r, base_identity=original)) for r in retained
                    ]
                if any(not torch.isfinite(torch.tensor(v)) or abs(v) > 1e-6 for v in initial_kl):
                    raise RuntimeError("Initial adapter does not match retention reference")
                write(
                    directory / "initial-retention.json",
                    {"kl": initial_kl, "forward_calls": len(retained)},
                )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
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
                loss_terms=loss_terms,
                preservation_objective=preservation_objective,
                retained_responses=retained,
                on_step=(lambda fitted, row: on_step(fitted, directory, row))
                if on_step is not None
                else None,
            )
            fit_seconds = time.monotonic() - started
            peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
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
        reloaded = load_lora(model, directory / "adapter", names, rank=rank, seed=protocol["seed"])
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
            "peak_training_allocated_bytes": peak_bytes,
            "adapter_bytes": sum(
                p.stat().st_size for p in (directory / "adapter").rglob("*") if p.is_file()
            ),
            "trainable_parameters": parameter_count,
            "successful_steps": len(history),
            "forward_calls": sum(
                r["forward_calls"] + sum(retry["forward_calls"] for retry in r["gradient_retries"])
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
            "retention_reference": retention_receipt,
            "arms": results,
        },
    )
    return model, results
