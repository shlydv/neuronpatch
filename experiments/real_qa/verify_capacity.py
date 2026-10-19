"""Verify that every broad-LoRA checkpoint reproduces its saved validation outputs."""

import gc
import json
from pathlib import Path

import torch
from data import MODEL, REVISION, prompt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    summary = json.loads((ROOT / "capacity/summary.json").read_text())
    checks = {}
    for name in summary["runs"]:
        base = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )
        model = PeftModel.from_pretrained(base, ROOT / "capacity" / name).eval()
        rows = json.loads((ROOT / "capacity" / f"{name}-val.json").read_text())
        mismatches = []
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
                texts = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                mismatches.extend(r["id"] for r, text in zip(pair, texts) if text != r["output"])
        checks[name] = {"examples": len(rows), "mismatches": mismatches}
        (ROOT / "capacity/reload_checks.json").write_text(json.dumps(checks, indent=2))
        if mismatches:
            raise AssertionError(f"Reload mismatch: {name}: {mismatches}")
        print(name, "PASS", flush=True)
        del model, base, outputs
        gc.collect()
        torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
