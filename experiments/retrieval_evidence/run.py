"""Bounded unchanged-model generation; correctness requires full-output review."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from neuronpatch.patch import fingerprint


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def execute(model, tokenizer, rows, protocol, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = fingerprint(model)
    write(output / "protocol.json", protocol)
    write(output / "inputs.json", rows)
    results, logits = [], {}
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if time.monotonic() - started > protocol["runtime_limit_seconds"]:
                raise TimeoutError("Bounded baseline runtime exceeded")
            prefix = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
            assert len(prefix) == row["prompt_tokens"]
            assert len(prefix) + protocol["max_new_tokens"] <= protocol["maximum_total_tokens"]
            ids = torch.tensor([prefix], device=model.device)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            first = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
            assert torch.isfinite(first).all()
            logits[row["id"]] = first.detach().cpu().contiguous()
            generated = (
                model.generate(
                    **inputs,
                    max_new_tokens=protocol["max_new_tokens"],
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[0, len(prefix) :]
                .cpu()
                .tolist()
            )
            ended = bool(generated and generated[-1] == tokenizer.eos_token_id)
            results.append(
                {
                    "id": row["id"],
                    "text": tokenizer.decode(generated, skip_special_tokens=True),
                    "tokens": generated,
                    "ended": ended,
                    "truncated": len(generated) == protocol["max_new_tokens"] and not ended,
                    "correctness": "UNREVIEWED",
                }
            )
            write(output / "predictions.json", results)
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"BASELINE {index + 1}/{len(rows)}", flush=True)
    save_file(logits, str(output / "first_logits.safetensors"))
    final = fingerprint(model)
    assert original == final
    write(
        output / "completion.json",
        {
            "status": "complete",
            "queries": len(results),
            "training_steps": 0,
            "initial_fingerprint": original,
            "final_fingerprint": final,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if model.device.type == "cuda"
            else 0,
        },
    )
    print("COMPLETE unchanged retrieval baseline", flush=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    raw = (args.data / "inputs.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol["inputs_sha256"]
    rows = json.loads(raw)
    assert len(rows) == protocol["queries"] and len({r["id"] for r in rows}) == len(rows)
    assert all(
        importlib.metadata.version(k).split("+")[0] == v for k, v in protocol["runtime"].items()
    )
    torch.set_num_threads(2)
    torch.manual_seed(43)
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
    execute(model, tokenizer, rows, protocol, args.output)


if __name__ == "__main__":
    main()
