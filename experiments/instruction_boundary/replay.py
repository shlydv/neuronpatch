"""Evaluate frozen comparator adapters without fitting or selecting them again."""

import hashlib
import json
from pathlib import Path

import torch

from experiments.paired_mlp.model import load_lora, merge_lora
from neuronpatch import NeuronPatch
from neuronpatch.patch import fingerprint

from .run import write
from .transfer_run import detach_grouped, generate, projection_names


def verify_parent(parent, manifest):
    """Validate every pinned artifact before loading any adapter."""
    parent = Path(parent)
    for name, expected in manifest["files_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe parent artifact path")
        if hashlib.sha256((parent / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Parent artifact hash mismatch: {name}")
    required = {"choices.json", "fits.json", "completion.json", "protocol.json"}
    if not required.issubset(manifest["files_sha256"]):
        raise ValueError("Parent identity files must be pinned")
    choices = json.loads((parent / "choices.json").read_text())
    if choices != manifest["choices"] or len(set(choices)) != len(choices):
        raise ValueError("Frozen parent choices differ")
    for label in choices:
        if Path(label).name != label:
            raise ValueError("Unsafe adapter label")
        directory = parent / f"{label}-adapter"
        actual = {str(p.relative_to(parent)) for p in directory.rglob("*") if p.is_file()}
        pinned = {name for name in manifest["files_sha256"] if name.startswith(f"{label}-adapter/")}
        if not actual or actual != pinned:
            raise ValueError("Every adapter file must be pinned")
    completion = json.loads((parent / "completion.json").read_text())
    if completion["status"] != "complete" or not completion["base_restored"]:
        raise ValueError("Incomplete parent experiment")
    if completion["original_fingerprint"] != manifest["base_fingerprint"]:
        raise ValueError("Parent base fingerprint mismatch")


def find_parent(root, manifest):
    expected = manifest["files_sha256"]["completion.json"]
    matches = [
        p.parent
        for p in Path(root).rglob("completion.json")
        if hashlib.sha256(p.read_bytes()).hexdigest() == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one frozen parent, found {len(matches)}")
    verify_parent(matches[0], manifest)
    return matches[0]


def replay(model, tokenizer, parent, manifest, rows, spec, output, check_time):
    """Merge each saved endpoint, generate, then restore the exact base weights.

    No optimizer or selection is created. Restoring full affected projections
    intentionally avoids trusting a separate support file for grouped adapters.
    """
    verify_parent(parent, manifest)
    original = fingerprint(model)
    if original != manifest["base_fingerprint"]:
        raise ValueError("Comparator base model differs")
    parent = Path(parent)
    fits = json.loads((parent / "fits.json").read_text())
    parent_spec = json.loads((parent / "protocol.json").read_text())
    grouped, lora = projection_names(model)
    receipts = []
    for label in manifest["choices"]:
        check_time()
        candidates = [row for row in fits if row["id"] == label]
        if len(candidates) != 1:
            raise ValueError("Parent choice must identify one fit")
        arm = candidates[0]["arm"]
        names = lora if arm["family"] == "lora" else grouped
        backups = {name: model.get_submodule(name).weight.detach().cpu().clone() for name in names}
        directory = parent / f"{label}-adapter"
        if arm["family"] == "lora":
            fitted = load_lora(
                model, directory, lora, rank=parent_spec["lora"]["rank"], seed=arm["seed"]
            )
            model = merge_lora(fitted)
        else:
            owner = NeuronPatch.load(model, directory)
            model = detach_grouped(owner, merge=True)
        model.eval().requires_grad_(False)
        try:
            generate(
                model,
                tokenizer,
                rows,
                spec,
                output / f"reused-{label}-confirmation.json",
                check_time,
            )
        finally:
            with torch.no_grad():
                for name, weight in backups.items():
                    model.get_submodule(name).weight.copy_(weight)
            if fingerprint(model) != original:
                raise ValueError("Comparator replay failed base restoration")
        receipts.append({"id": label, "new_training_steps": 0, "base_restored": True})
        write(output / "reused-comparators.json", receipts)
    return receipts
