"""Baseline-only diagnostic: context reporting, science and abstention controls."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch.patch import fingerprint


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def decision(text, *, kind):
    """No broad substring rules: 'not unknown' must never count as abstention."""
    text = text.strip()
    if kind == "mcq":
        match = re.fullmatch(r"\(?([A-D])\)?[.!?)]?", text)
        return match.group(1) if match else None
    normalized = text.rstrip(".!?").upper()
    return normalized if normalized in ("UNKNOWN", "CONFLICT") else None


def messages(row):
    return row.get("messages") or [{"role": "user", "content": row["prompt"]}]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    source = args.data / "inputs.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["inputs_sha256"]
    rows = json.loads(source.read_text())
    assert len(rows) == protocol["input_count"] and len({r["id"] for r in rows}) == len(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", protocol)
    started = time.monotonic()
    print("LOAD unchanged pinned 4B, no training", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    base_hash = fingerprint(model)
    write(
        args.output / "identity.json",
        {
            "fingerprint": base_hash,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "model": protocol["model"],
            "revision": protocol["revision"],
        },
    )
    letters = {k: tokenizer.encode(k, add_special_tokens=False) for k in "ABCD"}
    assert all(len(ids) == 1 for ids in letters.values())
    output = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if time.monotonic() - started > protocol["runtime_limit_seconds"]:
                raise TimeoutError("Frozen baseline diagnostic runtime limit exceeded")
            text = tokenizer.apply_chat_template(
                messages(row), tokenize=False, add_generation_prompt=True
            )
            prefix = tokenizer.encode(text, add_special_tokens=False)
            assert len(prefix) == row["prompt_tokens"]
            assert len(prefix) + protocol["max_new_tokens"] <= protocol["maximum_total_tokens"]
            ids = torch.tensor([prefix], device=model.device)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            first = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            probabilities = first.softmax(-1)
            letter_logits = torch.stack([first[letters[k][0]] for k in "ABCD"])
            conditional = letter_logits.softmax(-1)
            generated = model.generate(
                **inputs,
                max_new_tokens=protocol["max_new_tokens"],
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            new_ids = generated[0, len(prefix) :].cpu().tolist()
            answer = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed = decision(answer, kind=row["kind"])
            result = {
                k: v for k, v in row.items() if k not in ("messages", "prompt", "prompt_tokens")
            }
            result.update(
                text=answer,
                parsed=parsed,
                correct=parsed == row["answer"],
                generated_token_ids=new_ids,
                prompt_tokens=len(prefix),
                truncated=len(new_ids) == protocol["max_new_tokens"]
                and new_ids[-1] != tokenizer.eos_token_id,
                letter_probabilities={k: float(probabilities[letters[k][0]]) for k in "ABCD"},
                conditional_letter_probabilities={
                    k: float(conditional[i]) for i, k in enumerate("ABCD")
                },
            )
            output.append(result)
            # Persist incrementally so any provider interruption preserves completed cases.
            write(args.output / "predictions.json", output)
            if (index + 1) % 24 == 0 or index + 1 == len(rows):
                print(
                    f"EVAL {index + 1}/{len(rows)} task={row['task']} frame={row['frame']}",
                    flush=True,
                )
    final_hash = fingerprint(model)
    assert final_hash == base_hash
    write(
        args.output / "completion.json",
        {
            "status": "all frozen baseline queries complete",
            "queries": len(output),
            "seconds": time.monotonic() - started,
            "weights_unchanged": True,
            "final_fingerprint": final_hash,
            "training_steps": 0,
        },
    )
    print("COMPLETE baseline diagnostic; weights unchanged", flush=True)


if __name__ == "__main__":
    main()
