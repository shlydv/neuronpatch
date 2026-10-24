"""Hash-bound training inputs for a fixed-support supervision ablation."""

import hashlib
import json
from pathlib import Path

from .matched_supervision import full_document_supervision


def load_assets(directory, manifest):
    required = {
        "supports.json",
        "teachers-train.json",
        "teachers-calibration.json",
        "parent-fits.json",
    }
    if set(manifest["files_sha256"]) != required:
        raise ValueError("Exact frozen training asset set required")
    result = {}
    for name, expected in manifest["files_sha256"].items():
        raw = (Path(directory) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen training asset differs: {name}")
        result[name] = json.loads(raw)
    return result


def supervision(assets, cases, bank, manifest):
    return full_document_supervision(
        cases,
        assets[f"teachers-{bank}.json"],
        bank=bank,
        allowed_documents=manifest["membership"][bank],
    )


def fixed_supports(assets, arms, widths, *, rows, groups):
    """Return row/group identities only; no learned delta is ever loaded."""
    supports = assets["supports.json"]
    keys = {f"{a['family']}-{a['seed']}" for a in arms if a["family"] != "lora"}
    if set(supports) != keys:
        raise ValueError("Frozen support arms differ")
    for support in supports.values():
        if set(support) != set(widths):
            raise ValueError("Frozen projection coverage differs")
        for name, value in support.items():
            if set(value) != {"indices", "groups"}:
                raise ValueError("Only support identities may be loaded")
            indices, labels = value["indices"], value["groups"]
            if (
                len(indices) != rows
                or len(set(indices)) != rows
                or any(type(i) is not int or not 0 <= i < widths[name] for i in indices)
                or len(labels) != rows
                or any(type(i) is not int for i in labels)
                or set(labels) != set(range(groups))
            ):
                raise ValueError("Invalid frozen support or group identities")
    return supports


def verify_schedule(history, assets, label):
    matches = [row for row in assets["parent-fits.json"] if row["id"] == label]
    if len(matches) != 1:
        raise ValueError("Exactly one parent training schedule required")
    expected = [row["example_indices"] for row in matches[0]["history"]]
    actual = [row["example_indices"] for row in history]
    if actual != expected:
        raise ValueError("Supervision ablation changed sampled example indices")
    return {"parent_fit": label, "steps": len(actual), "sample_indices_exact": True}
