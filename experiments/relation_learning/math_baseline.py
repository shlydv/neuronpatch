"""Inference-only qualification of the frozen math development slice."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from decimal import Decimal
from pathlib import Path

import torch

from experiments.relation_learning.prepare_math import final_number


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def execute(model, tokenizer, rows, protocol, output, check_time):
    output.mkdir(parents=True, exist_ok=False)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or any(type(token) is not int for token in eos):
        raise ValueError("Explicit model termination tokens required")
    results = []
    for row in rows:
        check_time()
        ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        if ids.shape[1] + protocol["max_new_tokens"] > protocol["max_context_tokens"]:
            raise ValueError("Prompt exceeds frozen context limit")
        start = time.monotonic()
        sequence = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=protocol["max_new_tokens"],
            eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id or eos[0],
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
        )
        tokens = sequence[0, ids.shape[1] :].tolist()
        if not tokens or len(tokens) > protocol["max_new_tokens"]:
            raise ValueError("Unexpected generated sequence length")
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        number = final_number(text)
        ended = tokens[-1] in eos
        classification = (
            "incomplete"
            if not ended
            else "needs_review"
            if number is None
            else "numeric_correct"
            if Decimal(number) == Decimal(row["final_number"])
            else "numeric_wrong"
        )
        results.append(
            {
                "id": row["id"],
                "prompt_ids": ids[0].tolist(),
                "tokens": tokens,
                "text": text,
                "ended": ended,
                "parsed_number": number,
                "classification": classification,
                "seconds": time.monotonic() - start,
            }
        )
        write(output / "predictions.json", results)
        print("BASELINE", len(results), row["id"][:12], classification, flush=True)
    write(
        output / "completion.json",
        {
            "status": "complete",
            "examples": len(results),
            "training_updates": 0,
            "eos_token_ids": eos,
            "counts": {
                key: sum(r["classification"] == key for r in results)
                for key in ("numeric_correct", "numeric_wrong", "needs_review", "incomplete")
            },
            "decision": "pending full response review; no automatic training",
        },
    )
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    runtime = {k: importlib.metadata.version(k).split("+")[0] for k in protocol["runtime"]}
    if runtime != protocol["runtime"]:
        raise ValueError("Runtime differs from frozen dependencies")
    raw = (args.data / "inputs.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["inputs_sha256"]:
        raise ValueError("Input hash mismatch")
    rows = json.loads(raw)["splits"]["development"]
    if len(rows) != 128 or len({row["id"] for row in rows}) != 128:
        raise ValueError("Expected all 128 distinct development questions")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Baseline runtime limit reached")

    torch.set_num_threads(2)
    torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("STAGE load", protocol["model"], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    execute(model, tokenizer, rows, protocol, args.output, check_time)
    print("COMPLETE math-baseline", flush=True)


if __name__ == "__main__":
    main()
