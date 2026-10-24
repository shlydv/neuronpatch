"""Verify a complete preference pilot before inspecting semantic results.

Artifact consistency is not independent GPU replication or a correctness judge.
Tokenizer must be loaded from the frozen revision by the caller.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .preference_training import prepare_pair, reference_lookup


def read(path):
    return json.loads(Path(path).read_text())


def verify(root, data, spec, tokenizer):
    root = Path(root)
    if read(root / "protocol.json") != spec:
        raise ValueError("Worker protocol changed")
    if read(root / "complete.json") != {"conditions": 5, "base_restored": True}:
        raise ValueError("Worker did not complete all conditions")
    comparison = root / "comparison"
    complete = read(comparison / "complete.json")
    if complete["base_identity"] != spec["base_fingerprint"] or set(complete["arms"]) != set(
        spec["arms"]
    ):
        raise ValueError("Wrong base or incomplete arms")
    cache = read(comparison / "references.json")
    protocol = read(comparison / "protocol.json")
    if protocol["training"] != spec["training"] or protocol["rank"] != spec["rank"]:
        raise ValueError("Training settings changed")
    banks = {
        k: [
            prepare_pair(tokenizer, r, max_length=spec["max_sequence_tokens"], device="cpu")
            for r in data["train"][k + "_pairs"]
        ]
        for k in ("narrow", "diverse")
    }
    for bank, pairs in banks.items():
        reference_lookup(pairs, cache, base_identity=spec["base_fingerprint"])
        if protocol["banks"][bank] != [{"id": p.id, "identity": p.identity} for p in pairs]:
            raise ValueError("Training bank differs from frozen data")
    expected_ref = {p.id for pairs in banks.values() for p in pairs}
    if {r["id"] for r in cache["records"]} != expected_ref:
        raise ValueError("Unexpected reference data")
    if complete["reference_forward_calls"] != 2 * len(expected_ref):
        raise ValueError("Reference cost mismatch")
    cases = data["confirmation"]["cases"]
    outputs = {}
    for name in ["base", *spec["arms"]]:
        directory = root / "base" if name == "base" else comparison / name
        rows = read(directory / "confirmation.json")
        if [r["id"] for r in rows] != [r["id"] for r in cases]:
            raise ValueError("Generation coverage/order mismatch")
        for row, case in zip(rows, cases, strict=True):
            prompt = tokenizer.apply_chat_template(
                case["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if (
                row["prompt_ids"] != prompt
                or not 1 <= len(row["tokens"]) <= spec["generation"]["max_new_tokens"]
            ):
                raise ValueError("Wrong prompt or invalid generated token count")
            if (
                row["text"] != tokenizer.decode(row["tokens"], skip_special_tokens=True)
                or type(row["ended"]) is not bool
            ):
                raise ValueError("Output decoding or termination flag invalid")
        outputs[name] = rows
        if name == "base":
            continue
        coverage, objective = name.split("-")
        history = read(directory / "history.json")
        if len(history) != spec["training"]["steps"]:
            raise ValueError("Incomplete training")
        generator = torch.Generator().manual_seed(spec["training"]["seed"])
        orders = [
            torch.randperm(n, generator=generator).tolist()
            for n in (len(banks[coverage]), len(data["train"]["clean"]))
        ]
        calls = 0
        for step, row in enumerate(history):
            indices = [order[step % len(order)] for order in orders]
            if (
                row["step"] != step + 1
                or row["example_indices"] != indices
                or row["objective"] != objective
            ):
                raise ValueError("Optimizer schedule changed")
            if row["pair_id"] != banks[coverage][indices[0]].id:
                raise ValueError("Wrong training pair")
            expected_calls = 2 if objective == "sft" else 5
            if row["forward_calls"] != expected_calls:
                raise ValueError("Wrong forward cost")
            calls += row["forward_calls"] + sum(r["forward_calls"] for r in row["gradient_retries"])
        result = read(directory / "result.json")
        if result != complete["arms"][name] or result["forward_calls"] != calls:
            raise ValueError("Result manifest mismatch")
        if not result["base_restored"] or not result["export_score_exact"]:
            raise ValueError("Adapter lifecycle failed")
        tensors = load_file(directory / "adapter/adapter_model.safetensors")
        if not tensors or any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("Invalid exported parameters")
        if sum(t.numel() for t in tensors.values()) != result["trainable_parameters"]:
            raise ValueError("Exported parameter count mismatch")
    return {
        "conditions": len(outputs),
        "generations": sum(map(len, outputs.values())),
        "fits": len(spec["arms"]),
        "scope": "Artifact consistency, not independent GPU replication",
    }, outputs
