"""Replace short direct controls with full-document authority counterexamples.

This is an experimental data ablation, not a new neuron-selection algorithm.
Keep the task bank and preservation-bank size/order unchanged. Only the direct
redirect input is replaced; its marker target remains the same. Callers must
provide the frozen document membership for the requested training bank.
"""

from copy import deepcopy

from .authority_pairs import matched_authority_pairs
from .transfer_data import teacher_supervision


def full_document_supervision(cases, predictions, *, bank, allowed_documents):
    """Build train/calibration targets without admitting confirmation documents."""
    if bank not in ("train", "calibration"):
        raise ValueError("Only train/calibration supervision is permitted")
    allowed = set(allowed_documents)
    documents = {row["document_id"] for row in cases}
    if not allowed or documents != allowed:
        raise ValueError("Cases must exactly match frozen bank membership")
    original = teacher_supervision(cases, predictions)
    result = deepcopy(original)
    outside = {
        row["document_id"]: row
        for row in matched_authority_pairs(cases)
        if row["authority"] == "user"
    }
    replacements = []
    for row in result["preserve"]:
        if row["id"].endswith(":direct:redirect"):
            old_id = row["id"]
            row["id"] = row["document_id"] + ":direct_full:redirect"
            row["messages"] = deepcopy(outside[row["document_id"]]["messages"])
            replacements.append({"old_id": old_id, "new_id": row["id"]})
    if len(replacements) != len(result["task"]):
        raise ValueError("Exactly one full-document control per eligible teacher required")
    return result, {
        "bank": bank,
        "allowed_documents": sorted(allowed),
        "replacement_count": len(replacements),
        "replacements": replacements,
        "task_unchanged": result["task"] == original["task"],
        "preservation_examples": len(result["preserve"]),
        "compute_note": "Equal examples/steps does not imply equal token or compute cost",
    }
