"""Compare unchanged checkpoint inference; no adaptation or threshold search."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


def canonical_fp16_fingerprint(model):
    """Compare stored model values with the earlier FP16 baseline, streaming tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to(device="cpu")
        if value.is_floating_point():
            value = value.to(torch.float16)
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy()))
        del value
    return digest.hexdigest()


def generate(model, tokenizer, rows, maximum, prefix, logits):
    outputs = []
    with torch.inference_mode():
        for row in rows:
            tokens = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
            assert len(tokens) == row["prompt_tokens"]
            ids = torch.tensor([tokens], device=model.get_input_embeddings().weight.device)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            first = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
            assert torch.isfinite(first).all()
            logits[prefix + ":" + row["id"]] = first.detach().cpu().contiguous()
            result = (
                model.generate(
                    **inputs,
                    max_new_tokens=maximum,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[0, len(tokens) :]
                .cpu()
                .tolist()
            )
            outputs.append(
                {
                    "id": row["id"],
                    "tokens": result,
                    "text": tokenizer.decode(result, skip_special_tokens=True),
                    "ended": bool(result and result[-1] == tokenizer.eos_token_id),
                }
            )
            print("SANITY", prefix, row["id"], flush=True)
    return outputs


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
    assert len(rows) == protocol["queries_per_precision"]
    assert all(
        importlib.metadata.version(k).split("+")[0] == v for k, v in protocol["runtime"].items()
    )
    assert torch.cuda.device_count() == 2, "FP32 requires both free T4 GPUs; no CPU offload"
    torch.set_num_threads(2)
    torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs.json", "protocol.json"):
        (args.output / name).write_bytes((args.data / name).read_bytes())
    started = time.monotonic()
    outputs, logits, identities = {}, {}, {}
    for mode in protocol["modes"]:
        device_map = {"": 0} if mode == "fp16" else protocol["fp32_device_map"]
        model = (
            AutoModelForCausalLM.from_pretrained(
                protocol["model"],
                revision=protocol["revision"],
                torch_dtype=torch.float16 if mode.startswith("fp16") else torch.float32,
                device_map=device_map,
                attn_implementation="eager",
            )
            .eval()
            .requires_grad_(False)
        )
        actual_dtype = torch.float16 if mode.startswith("fp16") else torch.float32
        assert all(p.dtype == actual_dtype for p in model.parameters())
        identity = canonical_fp16_fingerprint(model)
        assert identity == protocol["base_fp16_fingerprint"]
        identities[mode] = {
            "canonical_fp16_fingerprint": identity,
            "hf_device_map": model.hf_device_map,
            "dtype": str(actual_dtype),
            "config": model.config.to_dict(),
        }
        outputs[mode] = generate(model, tokenizer, rows, protocol["max_new_tokens"], mode, logits)
        (args.output / "predictions.json").write_text(json.dumps(outputs, indent=2) + "\n")
        (args.output / "identities.json").write_text(json.dumps(identities, indent=2) + "\n")
        save_file(logits, str(args.output / "first_logits.safetensors"))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "training_steps": 0,
                "generations": sum(len(v) for v in outputs.values()),
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print("COMPLETE inference sanity", flush=True)


if __name__ == "__main__":
    main()
