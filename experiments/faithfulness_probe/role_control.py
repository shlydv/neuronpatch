"""Reconstruct the exploratory system-role inputs without changing instructions."""

import argparse
import hashlib
import json
from pathlib import Path

from .prepare import FRAMES, MODEL, REVISION


def system_control_rows(original, tokenizer):
    rows = []
    for row in original:
        if (
            row["task"] != "counterfactual"
            or row["order"] != "original"
            or row["frame"] not in ("world_with_context", "context_explicit")
        ):
            continue
        instruction = FRAMES[row["frame"]]
        if not row["prompt"].startswith(instruction + "\n\n"):
            raise ValueError("Instruction bytes differ from frozen original")
        record = {k: v for k, v in row.items() if k not in ("prompt", "prompt_tokens")}
        record.update(
            id=row["id"] + ":system",
            original_id=row["id"],
            frame=row["frame"] + "_system",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": row["prompt"][len(instruction) + 2 :]},
            ],
        )
        record["prompt_tokens"] = len(
            tokenizer.apply_chat_template(record["messages"], add_generation_prompt=True)
        )
        if record["prompt_tokens"] + 1024 > 2048:
            raise ValueError("Role-control token budget exceeded")
        rows.append(record)
    return rows


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((Path(__file__).parent / "protocol_v1.json").read_text())
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == protocol["inputs_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rows = system_control_rows(json.loads(args.source.read_text()), tokenizer)
    assert len(rows) == 96
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
