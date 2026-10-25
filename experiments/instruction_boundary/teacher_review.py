"""Bind source-review decisions to exact teacher trajectories before distillation.

This checks review coverage and identity, not semantic correctness or checkpoint
provenance. Tokenizer/model identity checks remain the caller's responsibility.
Keep excluded examples in the receipt; never silently treat pending as rejected.
"""

import hashlib
import json
from copy import deepcopy


def review_identity(record):
    """Hash both the source task and full teacher, including prompt/response IDs."""
    payload = {key: record[key] for key in ("task", "teacher")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def approved_teachers(inventory, decisions):
    """Return approved records and an exclusion receipt after complete review.

    Decisions contain split, document_id, identity, verdict (accept/reject),
    rationale, and reviewer. Every inventory item needs exactly one decision.
    An acceptance certifies the complete response, including supporting claims,
    according to the reviewer; this function does not implement an answer judge.
    """
    records = {}
    document_splits = {}
    for split, entries in inventory.items():
        for record in entries:
            did = record["task"]["document_id"]
            key = (split, did)
            if key in records or did in document_splits:
                raise ValueError("duplicate document or cross-split overlap")
            if record["teacher"]["id"].split(":")[0] != did:
                raise ValueError("teacher/document identity mismatch")
            records[key] = record
            document_splits[did] = split
    if not records:
        raise ValueError("empty teacher inventory")
    checked = {}
    for decision in decisions:
        key = (decision["split"], decision["document_id"])
        if key not in records or key in checked:
            raise ValueError("unknown or duplicate review decision")
        if decision["identity"] != review_identity(records[key]):
            raise ValueError("review identity changed")
        if decision["verdict"] not in ("accept", "reject"):
            raise ValueError("review verdict must be accept or reject")
        if any(
            not isinstance(decision.get(field), str) or not decision[field].strip()
            for field in ("rationale", "reviewer")
        ):
            raise ValueError("review rationale and reviewer required")
        checked[key] = decision
    if checked.keys() != records.keys():
        raise ValueError(f"incomplete teacher review: {len(records) - len(checked)} pending")
    approved = {split: [] for split in inventory}
    for key, record in records.items():
        if checked[key]["verdict"] == "accept":
            approved[key[0]].append(deepcopy(record))
    return approved, deepcopy(list(checked.values()))
