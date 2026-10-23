"""Independently check saved baseline artifacts; does not judge answer quality."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root, frozen_inputs, frozen_protocol, tokenizer):
    import torch
    from safetensors.torch import load_file

    def read(path):
        return json.loads(path.read_text())

    rows, predictions = read(root / "inputs.json"), read(root / "predictions.json")
    protocol, completion = read(root / "protocol.json"), read(root / "completion.json")
    assert rows == read(frozen_inputs)
    assert protocol == read(frozen_protocol)
    assert hashlib.sha256(frozen_inputs.read_bytes()).hexdigest() == protocol["inputs_sha256"]
    assert completion["status"] == "complete" and completion["training_steps"] == 0
    assert completion["initial_fingerprint"] == completion["final_fingerprint"]
    assert len(predictions) == len(rows) == protocol["queries"] == completion["queries"]
    assert [p["id"] for p in predictions] == [r["id"] for r in rows]
    logits = load_file(root / "first_logits.safetensors")
    assert set(logits) == {r["id"] for r in rows}
    first_token_disagreements = []
    for row, prediction in zip(rows, predictions, strict=True):
        prefix = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True
        )
        assert len(prefix) == row["prompt_tokens"]
        tokens = prediction["tokens"]
        assert 0 < len(tokens) <= protocol["max_new_tokens"]
        assert len(prefix) + len(tokens) <= protocol["maximum_total_tokens"]
        assert tokenizer.decode(tokens, skip_special_tokens=True) == prediction["text"]
        ended = tokens[-1] == tokenizer.eos_token_id
        assert ended == prediction["ended"]
        assert prediction["truncated"] == (len(tokens) == protocol["max_new_tokens"] and not ended)
        first = logits[row["id"]]
        assert first.ndim == 1 and torch.isfinite(first).all()
        if first.argmax().item() != tokens[0]:
            # Separate forward/generate kernels can differ in floating point.
            # Report instead of hiding this evidence or assuming exact parity.
            first_token_disagreements.append(
                {"id": row["id"], "gap": float(first.max() - first[tokens[0]])}
            )
    return {
        "queries": len(rows),
        "source_inputs_and_protocol_match": True,
        "decoded_outputs_match": True,
        "base_fingerprint_unchanged": True,
        "first_token_disagreements": first_token_disagreements,
        "truncated": sum(p["truncated"] for p in predictions),
        "ended": sum(p["ended"] for p in predictions),
        "elapsed_seconds": completion["elapsed_seconds"],
        "answer_quality_reviewed": False,
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"], revision=protocol["revision"], local_files_only=True
    )
    result = verify(args.root, args.inputs, args.protocol, tokenizer)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
