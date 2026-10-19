"""Measure development splits only; confirmation stays unopened by inference."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from data import MODEL, REVISION, metrics, prompt, score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent


def main(scaled=False):
    out = ROOT / ("scaled_baseline" if scaled else "baseline")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
        )
        .to("mps")
        .eval()
    )
    summary = {
        "model": MODEL,
        "scaled": scaled,
        "revision": REVISION,
        "max_new_tokens": 32,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_helper_sha256": hashlib.sha256((ROOT / "data.py").read_bytes()).hexdigest(),
        "splits": {},
    }
    started = time.monotonic()
    for split in ("val", "train", "cal"):
        path = (
            ROOT / "scaled" / f"{split}_expanded.json"
            if scaled
            else ROOT / "prepared" / f"{split}.json"
        )
        rows = json.loads(path.read_text())
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(rows), 2):
                part = rows[start : start + 2]
                inputs = tokenizer(
                    [prompt(tokenizer, r) for r in part],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to("mps")
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                )
                continuations = outputs[:, inputs["input_ids"].shape[1] :]
                for row, ids in zip(part, continuations):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    predictions.append(
                        row
                        | {
                            "output": text,
                            **score(text, row["answers"]),
                            "hit_token_limit": len(ids) == 32 and tokenizer.eos_token_id not in ids,
                        }
                    )
                if start % (64 if scaled else 16) == 0:
                    print(split, start, "/", len(rows), flush=True)
                    if scaled:
                        torch.mps.empty_cache()
        (out / f"{split}.json").write_text(json.dumps(predictions, indent=2))
        summary["splits"][split] = {
            "metrics": metrics(predictions),
            "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "hit_token_limit": sum(r["hit_token_limit"] for r in predictions),
        }
        if scaled:
            old = json.loads((ROOT / "baseline" / f"{split}.json").read_text())
            predicted = {r["id"]: r["output"] for r in predictions}
            mismatches = [r["id"] for r in old if predicted[r["id"]] != r["output"]]
            summary["splits"][split]["original_prefix_output_mismatches"] = mismatches
            if mismatches:
                raise AssertionError(f"Base outputs changed on original prefix: {mismatches}")
            old_ids = {r["id"] for r in old}
            summary["splits"][split]["new_only_metrics"] = metrics(
                [r for r in predictions if r["id"] not in old_ids]
            )
        summary["elapsed_seconds"] = time.monotonic() - started
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(split, summary["splits"][split], flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled", action="store_true")
    main(scaled=parser.parse_args().scaled)
