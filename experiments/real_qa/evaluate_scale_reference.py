"""Evaluate existing 128-example, one-epoch fits on the common larger validation set."""

import gc
import hashlib
import json
from pathlib import Path

import torch
from data import MODEL, REVISION, metrics, prompt, score
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent


def main():
    out = ROOT / "scale_128"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    path = ROOT / "scaled/val_expanded.json"
    rows = json.loads(path.read_text())
    previous = json.loads((ROOT / "capacity/summary.json").read_text())
    summary = {
        "model": MODEL,
        "revision": REVISION,
        "training_examples": 128,
        "reused_checkpoints": True,
        "validation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runs": {},
    }
    for lr in (0.0001, 0.0003):
        name = f"lora-lr{lr}-kl1.0-epoch1"
        base = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )
        model = PeftModel.from_pretrained(base, ROOT / "capacity" / name).eval()
        predictions = []
        with torch.inference_mode():
            for i in range(0, len(rows), 2):
                pair = rows[i : i + 2]
                inputs = tokenizer(
                    [prompt(tokenizer, r) for r in pair],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to("mps")
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for row, ids in zip(pair, outputs[:, inputs["input_ids"].shape[1] :]):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    predictions.append(
                        row
                        | {
                            "output": text,
                            **score(text, row["answers"]),
                            "hit_token_limit": len(ids) == 32 and tokenizer.eos_token_id not in ids,
                        }
                    )
                if i % 64 == 0:
                    print(name, i, flush=True)
                    torch.mps.empty_cache()
        old = json.loads((ROOT / "capacity" / f"{name}-val.json").read_text())
        predicted = {r["id"]: r["output"] for r in predictions}
        assert all(predicted[r["id"]] == r["output"] for r in old)
        old_ids = {r["id"] for r in old}
        (out / f"{name}-val.json").write_text(json.dumps(predictions, indent=2))
        source_run = previous["runs"][name]
        summary["runs"][name] = {
            "lr": lr,
            "parameters": source_run["parameters"],
            "optimizer_steps_total": 64,
            "anchor_count": len(previous["anchors"]),
            "anchor_em": source_run["anchor_em"],
            "train": source_run["train"],
            "validation": metrics(predictions),
            "new_only_metrics": metrics([r for r in predictions if r["id"] not in old_ids]),
            "original_prefix_output_mismatches": [],
            "training_seconds": source_run["training_seconds"],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(name, summary["runs"][name], flush=True)
        del model, base, outputs
        gc.collect()
        torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
