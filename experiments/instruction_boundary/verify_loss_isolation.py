"""CPU artifact checks for the loss diagnostic, not independent GPU replication."""

import hashlib
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from neuronpatch.supervision import chat_response

from .loss_isolation_run import compare_replay, verify_prior
from .preference_training import prepare_pair, reference_lookup
from .response_retention import response_positions, sample_identity
from .verify_preference import read


def verify(root, data, spec, tokenizer, prior):
    root, prior = Path(root), Path(prior)
    retention = spec["arms"] == ["diverse-sft-kl"]
    expected_fits = 1 if retention else 2
    verify_prior(prior, spec["prior_files"])
    if read(root / "protocol.json") != spec:
        raise ValueError("Worker protocol changed")
    completed = read(root / "complete.json")
    if (
        completed["fits"] != expected_fits
        or not completed["exact_prior_output_replay"]
        or not completed["base_restored"]
    ):
        raise ValueError("Worker did not complete verified controls")
    all_cases = {r["id"]: r for r in data["confirmation"]["cases"]}
    cases = [all_cases[i] for i in spec["diagnostic_case_ids"]]
    clean = [r for r in cases if r["kind"] == "clean"]

    def outputs(path, expected_cases):
        rows = read(path)
        if [r["id"] for r in rows] != [r["id"] for r in expected_cases]:
            raise ValueError("Generation membership/order changed")
        for row, case in zip(rows, expected_cases, strict=True):
            prompt = tokenizer.apply_chat_template(
                case["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if (
                row["prompt_ids"] != prompt
                or not 1 <= len(row["tokens"]) <= spec["generation"]["max_new_tokens"]
                or len(prompt) + spec["generation"]["max_new_tokens"]
                > spec["generation"]["max_context_tokens"]
            ):
                raise ValueError("Invalid prompt or generation length")
            if (
                row["text"] != tokenizer.decode(row["tokens"], skip_special_tokens=True)
                or type(row["ended"]) is not bool
            ):
                raise ValueError("Invalid decoded output or termination flag")
        return rows

    replay = outputs(root / "combined-replay/confirmation.json", cases)
    compare_replay(replay, read(prior / "confirmation.json"), spec["diagnostic_case_ids"])
    comparison = root / "comparison"
    protocol = read(comparison / "protocol.json")
    complete = read(comparison / "complete.json")
    if (
        protocol["training"] != spec["training"]
        or protocol["rank"] != spec["rank"]
        or protocol["base_identity"] != spec["base_fingerprint"]
        or complete["base_identity"] != spec["base_fingerprint"]
        or set(complete["arms"]) != set(spec["arms"])
    ):
        raise ValueError("Comparison configuration mismatch")
    banks = {
        k: [
            prepare_pair(tokenizer, r, max_length=spec["max_sequence_tokens"], device="cpu")
            for r in data["train"][k + "_pairs"]
        ]
        for k in ("narrow", "diverse")
    }
    cache = read(comparison / "references.json")
    for k, pairs in banks.items():
        reference_lookup(pairs, cache, base_identity=spec["base_fingerprint"])
        if protocol["banks"][k] != [{"id": p.id, "identity": p.identity} for p in pairs]:
            raise ValueError("Training bank changed")
    reference_ids = {p.id for pairs in banks.values() for p in pairs}
    if {r["id"] for r in cache["records"]} != reference_ids or complete[
        "reference_forward_calls"
    ] != 2 * len(reference_ids):
        raise ValueError("Reference membership/cost differs")
    if retention:
        verify_retention_cache(comparison, data, spec, tokenizer, complete)
    final = {"combined": replay}
    intermediate = {}
    for name in spec["arms"]:
        directory = comparison / name
        mode = "combined" if retention else name.removeprefix("diverse-sft-")
        if mode not in ("task_only", "preservation_only", "combined"):
            raise ValueError("Unknown diagnostic arm")
        history = read(directory / "history.json")
        if len(history) != spec["training"]["steps"]:
            raise ValueError("Incomplete training")
        rng = torch.Generator().manual_seed(spec["training"]["seed"])
        orders = [
            torch.randperm(n, generator=rng).tolist()
            for n in (len(banks["diverse"]), len(data["train"]["clean"]))
        ]
        calls = 0
        for i, row in enumerate(history):
            indices = [o[i % len(o)] for o in orders]
            if (
                row["step"] != i + 1
                or row["example_indices"] != indices
                or row["pair_id"] != banks["diverse"][indices[0]].id
                or row["objective"] != "sft"
                or row["loss_terms"] != mode
            ):
                raise ValueError("Loss mode or training schedule changed")
            expected_calls = 2 if retention else 1
            active, inactive = (
                ("loss", "preserve_nll") if mode == "task_only" else ("preserve_nll", "loss")
            )
            if retention:
                active, inactive = "loss", "preserve_nll"
                if row.get("preservation_objective") != "kl" or not math.isfinite(
                    row["preserve_kl"]
                ):
                    raise ValueError("Invalid KL retention loss")
            if (
                row[inactive] is not None
                or not math.isfinite(row[active])
                or not math.isfinite(row["gradient_norm"])
                or row["forward_calls"] != expected_calls
            ):
                raise ValueError("Invalid active/inactive loss or gradient")
            for retry in row["gradient_retries"]:
                if retry["pair_id"] != row["pair_id"] or retry["forward_calls"] != expected_calls:
                    raise ValueError("Retry changed sample or cost")
            calls += expected_calls * (1 + len(row["gradient_retries"]))
        result = read(directory / "result.json")
        if (
            result != complete["arms"][name]
            or result["forward_calls"] != calls
            or result["successful_steps"] != len(history)
            or not result["base_restored"]
            or not result["export_score_exact"]
        ):
            raise ValueError("Training/export manifest mismatch")
        tensors = load_file(directory / "adapter/adapter_model.safetensors")
        if (
            not tensors
            or any(not torch.isfinite(t).all() for t in tensors.values())
            or sum(t.numel() for t in tensors.values()) != result["trainable_parameters"]
        ):
            raise ValueError("Invalid exported adapter")
        final[name] = outputs(directory / "confirmation.json", cases)
        for step in spec["diagnostic_steps"]:
            intermediate[f"{name}-step-{step}"] = outputs(
                directory / f"clean-step-{step}.json", clean
            )
    count = sum(map(len, final.values())) + sum(map(len, intermediate.values()))
    return (
        {
            "fits": expected_fits,
            "generations": count,
            "exact_prior_output_replay": True,
            "scope": (
                "Artifact consistency, not independent GPU replication; "
                "termination flag not independently recomputed"
            ),
        },
        final,
        intermediate,
    )


def verify_retention_cache(comparison, data, spec, tokenizer, complete):
    receipt = read(comparison / "retention-reference.json")
    path = comparison / "retention-reference.safetensors"
    samples = [
        chat_response(
            tokenizer,
            r["messages"],
            r["target"],
            max_length=spec["max_sequence_tokens"],
            device="cpu",
        )
        for r in data["train"]["clean"]
    ]
    if (
        receipt != complete["retention_reference"]
        or receipt["base_identity"] != spec["base_fingerprint"]
        or receipt["sample_sha256"] != [sample_identity(s) for s in samples]
        or receipt["file_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        or receipt["forward_calls"] != len(samples)
    ):
        raise ValueError("Retention cache identity mismatch")
    tensors = load_file(path)
    if set(tensors) != {str(i) for i in range(len(samples))}:
        raise ValueError("Retention cache membership mismatch")
    vocabulary_sizes = set()
    for i, sample in enumerate(samples):
        tensor = tensors[str(i)]
        if (
            tensor.ndim != 3
            or tensor.shape[:2] != (1, len(response_positions(sample)))
            or tensor.dtype != torch.float16
            or not torch.isfinite(tensor).all()
            or tensor.shape[-1] < 1
        ):
            raise ValueError("Invalid retention reference tensor")
        vocabulary_sizes.add(tensor.shape[-1])
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    if (
        len(vocabulary_sizes) != 1
        or size != receipt["tensor_bytes"]
        or size > spec["max_retention_cache_bytes"]
    ):
        raise ValueError("Retention cache size mismatch")
    initial = read(comparison / "diverse-sft-kl/initial-retention.json")
    if (
        initial["forward_calls"] != len(samples)
        or len(initial["kl"]) != len(samples)
        or any(not math.isfinite(v) or abs(v) > 1e-6 for v in initial["kl"])
    ):
        raise ValueError("Initial retention distribution drift")
