"""Route competing-task responses to the correct source for blinded QA review.

Use only after artifact verification. Trusted replacements change the target
question; untrusted replacements do not. Original case IDs remain unchanged for
paired analysis. The existing reviewer must reject unsupported additional answers,
not merely find a correct substring. No model labels enter the review packet.
"""

import copy

from .review_packet import build, validate_review


def review_cases(cases):
    clean = {c["document_id"]: c for c in cases if c["kind"] == "clean"}
    result = []
    for original in cases:
        case = copy.deepcopy(original)
        if case["kind"] in ("cross_direct", "cross_injected"):
            target = (
                case["donor_document_id"] if case["kind"] == "cross_direct" else case["document_id"]
            )
            if target not in clean or case["expected"] != clean[target]["expected"]:
                raise ValueError("Competing-task target differs from reviewed source answer")
            case["document_id"] = target
            case["kind"] = "injected"
        result.append(case)
    return result


def packet(cases, outputs):
    return build(review_cases(cases), outputs)


def apply(packet, key, review, cases, outputs):
    return validate_review(packet, key, review, review_cases(cases), outputs)
