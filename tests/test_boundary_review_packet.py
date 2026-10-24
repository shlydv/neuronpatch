import copy

import pytest

from experiments.instruction_boundary.prepare import examples
from experiments.instruction_boundary.review_packet import build, digest, validate_review


def fixture():
    cases = examples({"context": "A: 2", "question": "A?", "ideal": "2"}, "doc")
    outputs = {}
    for name in ("base", "candidate"):
        outputs[name] = [
            {
                "id": row["id"],
                "text": "2" if row["kind"] in ("clean", "injected") else row["marker"],
                "ended": True,
            }
            for row in cases
        ]
    return cases, outputs


def review_for(packet):
    return {
        "packet_sha256": digest(packet),
        "documents": {
            d["document_id"]: {
                "exclude_qa": False,
                "reason": "Unambiguous two-column table.",
                "responses": {
                    r["response_id"]: {"correct": True, "reason": "Matches table."}
                    for r in d["responses"]
                },
            }
            for d in packet
        },
    }


def test_review_is_arm_blinded_deduplicated_and_bound_to_outputs():
    cases, outputs = fixture()
    packet, key = build(cases, outputs)
    assert len(packet) == 1 and len(packet[0]["responses"]) == 1
    swapped = {"base": outputs["candidate"], "unknown_arm": outputs["base"]}
    assert build(cases, swapped)[0] == packet
    assert len(key["exact_controls"]) == 8
    review = review_for(packet)
    scored = validate_review(packet, key, review, cases, outputs)
    assert len(scored) == 14 and all(r["correct"] for r in scored)
    changed = copy.deepcopy(outputs)
    changed["candidate"][0]["text"] = "3"
    with pytest.raises(ValueError, match="bound"):
        validate_review(packet, key, review, cases, changed)
    review["documents"]["doc"]["responses"] = {}
    with pytest.raises(ValueError, match="every unique"):
        validate_review(packet, key, review, cases, outputs)


def test_incomplete_cannot_pass_and_ambiguity_excludes_qa_for_every_arm_only():
    cases, outputs = fixture()
    outputs["candidate"][0]["ended"] = False
    packet, key = build(cases, outputs)
    review = review_for(packet)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_review(packet, key, review, cases, outputs)
    for response in packet[0]["responses"]:
        if not response["ended"]:
            review["documents"]["doc"]["responses"][response["response_id"]]["correct"] = False
    review["documents"]["doc"]["exclude_qa"] = True
    scored = validate_review(packet, key, review, cases, outputs)
    assert sum(row["excluded"] for row in scored) == 6
    assert all(not row["excluded"] for row in scored if "response_id" not in row)
    outputs["base"] = outputs["base"][:-1]
    with pytest.raises(ValueError, match="cover each case"):
        build(cases, outputs)
