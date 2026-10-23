"""Recompute baseline decoding and numerical scoring without importing its worker."""

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path


def marked_value(text):
    # Independently enforce a unique marker beginning the final answer line.
    if text.count("####") != 1:
        return None
    match = re.search(
        r"(?m)^####\s*([+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?)\s*\Z", text
    )
    return None if match is None else Decimal(match[1].replace(",", ""))


def verify(data, protocol, predictions, completion, tokenizer, eos):
    rows = data["splits"]["development"]
    if len(rows) != 128 or len(predictions) != len(rows):
        raise ValueError("The full frozen 128-example result is required")
    if completion.get("status") != "complete" or completion.get("training_updates") != 0:
        raise ValueError("Expected completed inference-only execution")
    if completion["eos_token_ids"] != eos or completion["examples"] != len(rows):
        raise ValueError("Completion metadata differs from pinned model or frozen data")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate development IDs")
    counts = Counter()
    review = []
    vocabulary_size = len(tokenizer)
    for row, result in zip(rows, predictions, strict=True):
        if row["id"] != result["id"]:
            raise ValueError("Missing, repeated or reordered result")
        expected_prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True
        )
        if result["prompt_ids"] != expected_prompt:
            raise ValueError("Prompt token mismatch")
        tokens = result["tokens"]
        if (
            not tokens
            or len(tokens) > protocol["max_new_tokens"]
            or any(type(t) is not int or not 0 <= t < vocabulary_size for t in tokens)
        ):
            raise ValueError("Invalid or over-budget output tokens")
        if any(t in eos for t in tokens[:-1]):
            raise ValueError("Output continues after a termination token")
        if tokenizer.decode(tokens, skip_special_tokens=True) != result["text"]:
            raise ValueError("Decoded text differs from saved tokens")
        ended = tokens[-1] in eos
        if result["ended"] != ended:
            raise ValueError("Termination flag mismatch")
        value = marked_value(result["text"])
        stored = result["parsed_number"]
        if (stored is None) != (value is None) or (stored is not None and Decimal(stored) != value):
            raise ValueError("Parsed number mismatch")
        reference = marked_value(row["response"])
        if reference is None or reference != Decimal(row["final_number"]):
            raise ValueError("Reference number mismatch")
        label = (
            "incomplete"
            if not ended
            else "needs_review"
            if value is None
            else "numeric_correct"
            if value == reference
            else "numeric_wrong"
        )
        if result["classification"] != label:
            raise ValueError("Classification mismatch")
        counts[label] += 1
        if label != "numeric_correct":
            review.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "reference": row["response"],
                    "response": result["text"],
                    "response_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
                    "classification": label,
                }
            )
    expected = {
        k: counts[k] for k in ("numeric_correct", "numeric_wrong", "needs_review", "incomplete")
    }
    if completion["counts"] != expected:
        raise ValueError("Reported totals do not reproduce")
    return {
        "verified_examples": len(rows),
        "counts": expected,
        "review_queue": review,
        "qualification": "requires mathematical review; not decided by token verification",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.data / "inputs.json").read_bytes()
    protocol = json.loads((args.data / "protocol.json").read_text())
    if hashlib.sha256(raw).hexdigest() != protocol["inputs_sha256"]:
        raise ValueError("Frozen input hash mismatch")
    from transformers import AutoTokenizer, GenerationConfig

    kwargs = {"revision": protocol["revision"], "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], **kwargs)
    eos = GenerationConfig.from_pretrained(protocol["model"], **kwargs).eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    result = verify(
        json.loads(raw),
        protocol,
        json.loads((args.results / "predictions.json").read_text()),
        json.loads((args.results / "completion.json").read_text()),
        tokenizer,
        eos,
    )
    result["inputs_sha256"] = protocol["inputs_sha256"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
