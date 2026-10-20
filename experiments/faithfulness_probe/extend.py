"""Complete ALL token-limit-truncated answers without changing inputs or weights.

This is a separately labeled evaluation repair. Original frozen-run predictions
and strict metrics are retained. Every extended sequence must reproduce the exact
original generated prefix before providing additional tokens.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch.patch import fingerprint

from .run import messages, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    protocol = json.loads((args.original / "protocol.json").read_text())
    predictions = json.loads((args.original / "predictions.json").read_text())
    selected = [r for r in predictions if r["truncated"]]
    assert [r["id"] for r in selected] == spec["ids"]
    assert (
        hashlib.sha256((args.data / "inputs.json").read_bytes()).hexdigest()
        == protocol["inputs_sha256"]
    )
    source = {r["id"]: r for r in json.loads((args.data / "inputs.json").read_text())}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"],
        revision=protocol["revision"],
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)
    identity = fingerprint(model)
    assert (
        identity == json.loads((args.original / "completion.json").read_text())["final_fingerprint"]
    )
    results = []
    with torch.inference_mode():
        for i, old in enumerate(selected):
            if time.monotonic() - started > spec["runtime_limit_seconds"]:
                raise TimeoutError("Extension runtime limit exceeded")
            row = source[old["id"]]
            text = tokenizer.apply_chat_template(
                messages(row), tokenize=False, add_generation_prompt=True
            )
            prefix = tokenizer.encode(text, add_special_tokens=False)
            ids = torch.tensor([prefix], device=model.device)
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=spec["max_new_tokens"],
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            tokens = generated[0, len(prefix) :].cpu().tolist()
            assert tokens[: len(old["generated_token_ids"])] == old["generated_token_ids"], old[
                "id"
            ]
            results.append(
                {
                    "id": old["id"],
                    "original_text": old["text"],
                    "text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "generated_token_ids": tokens,
                    "original_prefix_exact": True,
                    "truncated": len(tokens) == spec["max_new_tokens"]
                    and tokens[-1] != tokenizer.eos_token_id,
                }
            )
            write(args.output / "predictions.json", results)
            print(f"EXTENDED {i + 1}/{len(selected)} id={old['id']}", flush=True)
    assert fingerprint(model) == identity
    write(
        args.output / "completion.json",
        {
            "status": "all originally truncated answers extended",
            "queries": len(results),
            "seconds": time.monotonic() - started,
            "weights_unchanged": True,
            "all_original_prefixes_exact": True,
            "still_truncated": sum(r["truncated"] for r in results),
        },
    )


if __name__ == "__main__":
    main()
