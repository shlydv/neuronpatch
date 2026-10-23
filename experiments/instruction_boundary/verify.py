"""Independent token/artifact checks; semantic QA judgments remain explicit."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(rows, predictions, completion, tokenizer, eos, protocol):
    if len(rows) != protocol["expected_examples"] or len(predictions) != len(rows):
        raise ValueError("Full frozen baseline coverage required")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    if (
        completion["status"] != "complete"
        or completion["training_updates"] != 0
        or completion["examples"] != len(rows)
        or completion["eos_token_ids"] != eos
    ):
        raise ValueError("Completion metadata mismatch")
    size = len(tokenizer)
    counts = {"ended": 0, "literal_match": 0, "marker_present": 0}
    for row, output in zip(rows, predictions, strict=True):
        if row["id"] != output["id"]:
            raise ValueError("Result order/identity mismatch")
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True
        )
        if prompt != output["prompt_ids"]:
            raise ValueError("Prompt token mismatch")
        if len(prompt) + protocol["max_new_tokens"] > protocol["max_context_tokens"]:
            raise ValueError("Frozen context budget exceeded")
        tokens = output["tokens"]
        if (
            not tokens
            or len(tokens) > protocol["max_new_tokens"]
            or any(type(t) is not int or not 0 <= t < size for t in tokens)
            or any(t in eos for t in tokens[:-1])
        ):
            raise ValueError("Invalid or over-budget generated tokens")
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if text != output["text"]:
            raise ValueError("Decoded text mismatch")
        flags = {
            "ended": tokens[-1] in eos,
            "literal_match": tokens[-1] in eos and text.strip() == row["expected"],
            "marker_present": row["marker"] in text,
        }
        for key, value in flags.items():
            if type(output[key]) is not bool or output[key] != value:
                raise ValueError("Reported flag mismatch")
            counts[key] += value
    if completion["counts"] != counts:
        raise ValueError("Summary does not reproduce")
    return {
        "verified_examples": len(rows),
        "counts": counts,
        "decision": "Semantic review and paired qualification still required",
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
        raise ValueError("Frozen inputs differ")
    from transformers import AutoTokenizer, GenerationConfig

    kwargs = {"revision": protocol["revision"], "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], **kwargs)
    eos = GenerationConfig.from_pretrained(protocol["model"], **kwargs).eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    result = verify(
        json.loads(raw)["splits"]["development"],
        json.loads((args.results / "predictions.json").read_text()),
        json.loads((args.results / "completion.json").read_text()),
        tokenizer,
        eos,
        protocol,
    )
    result["inputs_sha256"] = protocol["inputs_sha256"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
