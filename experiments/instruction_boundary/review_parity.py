"""Replay prior response-bound judgments through the new review pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.instruction_boundary.paired_results import summarize
from experiments.instruction_boundary.review_packet import build, digest, validate_review


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.legacy_root
    results = root / "download/boundary-transfer-results/results"
    cases = json.loads((root / "data/inputs.json").read_text())["splits"]["confirmation"]
    conditions = ["base", *json.loads((results / "choices.json").read_text())]
    outputs = {c: json.loads((results / f"{c}-confirmation.json").read_text()) for c in conditions}
    packet, key = build(cases, outputs)
    byid = {r["id"]: r for r in cases}
    old = json.loads((root / "semantic_review.json").read_text())
    report = {}
    for mode, legacy in old.items():
        lookup = {(r["model"], r["id"]): r for r in legacy["records"]}
        index = {byid[r["id"]]["document_id"]: r["document_index"] for r in legacy["records"]}
        review = {"packet_sha256": digest(packet), "documents": {}}
        for doc in packet:
            clean_id = doc["document_id"] + ":clean:none"
            reference = lookup["base", clean_id]
            review["documents"][doc["document_id"]] = {
                "exclude_qa": reference["excluded"],
                "reason": "Preserve prior reviewed document exclusion.",
                "responses": {},
            }
        response_docs = {r["response_id"]: d["document_id"] for d in packet for r in d["responses"]}
        for entry in key["attribution"]:
            previous = lookup[entry["condition"], entry["case_id"]]
            original = next(r for r in outputs[entry["condition"]] if r["id"] == entry["case_id"])
            assert (
                previous["response_sha256"] == hashlib.sha256(original["text"].encode()).hexdigest()
            )
            dst = review["documents"][response_docs[entry["response_id"]]]["responses"]
            decision = {"correct": previous["correct"], "reason": previous["basis"]}
            if entry["response_id"] in dst:
                assert dst[entry["response_id"]]["correct"] == decision["correct"]
            else:
                dst[entry["response_id"]] = decision
        records = validate_review(packet, key, review, cases, outputs)
        assert len(records) == len(legacy["records"]) == 1680
        for r in records:
            prev = lookup[r["condition"], r["case_id"]]
            assert all(r[k] == prev[k] for k in ("correct", "excluded", "ended"))
        new = summarize(cases, records)
        for previous in legacy["summary"]:
            assert (
                sorted(index[d] for d in new["baseline_clean_correct_documents"])
                == previous["base_clean_correct_documents"]
            )
            for group, values in new["conditions"][previous["model"]].items():
                expected = previous[group]
                for field in ("count", "correct", "incomplete"):
                    assert values[field] == expected[field]
                for field in ("corrections", "regressions"):
                    assert sorted(index[d] for d in values[field]) == sorted(expected[field])
                if group.startswith("injected:"):
                    assert sorted(index[d] for d in values["recovered"]) == sorted(
                        expected["paired_recovered"]
                    )
        report[mode] = {
            "reviewed_records": len(records),
            "condition_groups": sum(len(v) for v in new["conditions"].values()),
            "legacy_counts_and_document_ids_exact": True,
        }
    public = {
        "scope": (
            "New blinded-review and paired-summary pipeline replays existing judgments exactly; "
            "not independent semantic re-review or model reproduction."
        ),
        "legacy_review_sha256": hashlib.sha256(
            (root / "semantic_review.json").read_bytes()
        ).hexdigest(),
        "packet_sha256": digest(packet),
        "modes": report,
    }
    args.output.write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(public))


if __name__ == "__main__":
    main()
