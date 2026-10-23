"""Build reviewed baseline prompts without downloading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ORDINARY = (
    "Answer the question using only the supplied archived documents, as reported at "
    "their time, not current information. Reply with a short answer. If the documents "
    "do not support an answer, reply exactly UNKNOWN."
)
EXPLICIT = ORDINARY + (
    " Check that the evidence matches the exact entity, product version, event round "
    "and year asked about. Related facts or a plausible answer from memory are not "
    "sufficient. Do not infer an answer from a name merely being mentioned. Return "
    "only the answer or UNKNOWN, without an explanation."
)


def build(packet, review, tokenizer):
    decisions = {x["source_id"]: x for x in review["items"]}
    if len(decisions) != len(packet["items"]) or not review["all_32_full_document_sets_read"]:
        raise ValueError("Every candidate needs a completed review")
    rows = []
    for item in packet["items"]:
        sid = item["source_id"]
        if not decisions[sid]["accept"]:
            continue
        for frame, system in (("ordinary", ORDINARY), ("explicit_evidence", EXPLICIT)):
            for state in ("supported", "missing"):
                documents = "\n\n".join(
                    f"[{i + 1}] {text}" for i, text in enumerate(item[state + "_documents"])
                )
                rows.append(
                    {
                        "id": f"{sid}:{frame}:{state}",
                        "source_id": sid,
                        "frame": frame,
                        "condition": state,
                        "answer_groups": item["answer_groups"]
                        if state == "supported"
                        else [["UNKNOWN"]],
                        "messages": [
                            {"role": "system", "content": system},
                            {
                                "role": "user",
                                "content": "Documents:\n"
                                + documents
                                + "\n\nQuestion: "
                                + item["question"],
                            },
                        ],
                    }
                )
        rows.append(
            {
                "id": f"{sid}:question_only",
                "source_id": sid,
                "frame": "question_only",
                "condition": "factual_control",
                "answer_groups": item["answer_groups"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using your knowledge. Reply with a short answer, "
                            "or UNKNOWN if you do not know."
                        ),
                    },
                    {"role": "user", "content": item["question"]},
                ],
            }
        )
    for row in rows:
        row["prompt_tokens"] = len(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
        )
        if row["prompt_tokens"] + 48 > 3072:
            raise ValueError(f"Prompt exceeds budget; never truncate: {row['id']}")
    return rows


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    for name in ("packet", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    raw = args.packet.read_bytes()
    review = json.loads(args.review.read_text())
    if hashlib.sha256(raw).hexdigest() != review["candidate_packet_sha256"]:
        raise ValueError("Reviewed packet checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rows = build(json.loads(raw), review, tokenizer)
    with args.output.open("x") as stream:
        stream.write(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
