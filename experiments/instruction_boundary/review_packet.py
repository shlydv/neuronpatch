"""Build an arm-blinded semantic-review packet and a separate attribution key.

This is an experiment evaluation aid, not an automatic answer judge. Run the
artifact verifier first. Source documents and generated text remain private.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def build(cases, outputs):
    """Bind every response to its case while withholding arm names from reviewers."""
    case_ids = [row["id"] for row in cases]
    if len(set(case_ids)) != len(case_ids) or not outputs or "base" not in outputs:
        raise ValueError("Unique cases and a base condition required")
    indexed = {}
    for condition, rows in outputs.items():
        if len(rows) != len(cases) or {row["id"] for row in rows} != set(case_ids):
            raise ValueError("Every condition must cover each case exactly once")
        for row in rows:
            if not isinstance(row["text"], str) or type(row["ended"]) is not bool:
                raise ValueError("Text and explicit termination flag required")
        indexed[condition] = {row["id"]: row for row in rows}
    documents = {}
    attribution = []
    controls = []
    for case in cases:
        document_id = case["document_id"]
        if case["kind"] == "clean":
            if document_id in documents:
                raise ValueError("Exactly one clean question per document required")
            prefix, separator, encoded = case["messages"][-1]["content"].partition(
                "\n\nQuoted document (JSON string):\n"
            )
            question_prefix = "Answer this question using the quoted document: "
            if not separator or not prefix.startswith(question_prefix):
                raise ValueError("Unknown clean prompt format")
            documents[document_id] = {
                "document_id": document_id,
                "question": prefix[len(question_prefix) :],
                "document": json.loads(encoded),
                "reference_answer": case["expected"],
                "responses": {},
            }
    for case in cases:
        if case["document_id"] not in documents:
            raise ValueError("Case has no clean source document")
        for condition in sorted(indexed):
            output = indexed[condition][case["id"]]
            identity = {
                "condition": condition,
                "case_id": case["id"],
                "output_sha256": digest(output),
                "ended": output["ended"],
            }
            if case["kind"] in ("direct", "quoted", "direct_full"):
                controls.append(
                    {
                        **identity,
                        "correct": output["ended"] and output["text"].strip() == case["expected"],
                    }
                )
            elif case["kind"] in ("clean", "injected"):
                response = {"text": output["text"], "ended": output["ended"]}
                response_id = digest({"document_id": case["document_id"], **response})
                documents[case["document_id"]]["responses"][response_id] = response
                attribution.append({**identity, "response_id": response_id})
            else:
                raise ValueError("Unknown evaluation case kind")
    packet = []
    for document_id in sorted(documents):
        document = documents[document_id]
        document["responses"] = [
            {"response_id": key, **value} for key, value in sorted(document["responses"].items())
        ]
        packet.append(document)
    return packet, {
        "packet_sha256": digest(packet),
        "cases_sha256": digest(cases),
        "outputs_sha256": digest(outputs),
        "attribution": attribution,
        "exact_controls": controls,
    }


def validate_review(packet, key, review, cases, outputs):
    """Reject incomplete/stale review; enforce document-wide QA exclusions."""
    actual_packet, actual_key = build(cases, outputs)
    if packet != actual_packet or key != actual_key or review["packet_sha256"] != digest(packet):
        raise ValueError("Review is not bound to these exact outputs")
    if set(review["documents"]) != {row["document_id"] for row in packet}:
        raise ValueError("Review must cover all documents")
    decisions = {}
    for document in packet:
        decision = review["documents"][document["document_id"]]
        if type(decision["exclude_qa"]) is not bool or not decision["reason"].strip():
            raise ValueError("Document exclusion flag and rationale required")
        if set(decision["responses"]) != {r["response_id"] for r in document["responses"]}:
            raise ValueError("Review must cover every unique QA response")
        for response in document["responses"]:
            item = decision["responses"][response["response_id"]]
            if type(item["correct"]) is not bool or not item["reason"].strip():
                raise ValueError("Explicit response correctness and rationale required")
            if item["correct"] and not response["ended"]:
                raise ValueError("Incomplete response cannot count as correct")
            decisions[response["response_id"]] = {
                **item,
                "excluded": decision["exclude_qa"],
            }
    return [{**entry, **decisions[entry["response_id"]]} for entry in key["attribution"]] + [
        {**entry, "excluded": False} for entry in key["exact_controls"]
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="Completed response-bound annotation file")
    args = parser.parse_args()
    data = json.loads((args.data / "inputs.json").read_text())
    spec = json.loads((args.data / "protocol.json").read_text())
    choices = json.loads((args.results / "choices.json").read_text())
    conditions = [
        "base",
        *choices,
        *["reused-" + label for label in spec.get("reused_parent", {}).get("choices", [])],
    ]
    outputs = {
        name: json.loads((args.results / f"{name}-confirmation.json").read_text())
        for name in conditions
    }
    packet, key = build(data["splits"]["confirmation"], outputs)
    reviewed = None
    if args.review is not None:
        reviewed = validate_review(
            packet,
            key,
            json.loads(args.review.read_text()),
            data["splits"]["confirmation"],
            outputs,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("review-packet.json", packet), ("attribution-key.json", key)):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    if reviewed is not None:
        from .paired_results import summarize

        summary = summarize(data["splits"]["confirmation"], reviewed)
        for name, value in (("reviewed-records.json", reviewed), ("paired-summary.json", summary)):
            (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "documents": len(packet),
                "unique_qa_responses": sum(len(d["responses"]) for d in packet),
            }
        )
    )


if __name__ == "__main__":
    main()
