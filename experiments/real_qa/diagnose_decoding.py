"""Bounded decoding audit: same checkpoint, default versus neutral repetition penalty."""

import json
from pathlib import Path

import torch
from data import MODEL, REVISION, prompt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    name = "lora-lr0.0003-kl1.0-epoch1"
    diagnostic = json.loads((ROOT / "coverage_512/first_token_diagnostic.json").read_text())
    values = diagnostic["runs"][f"coverage_512/{name}"]
    mismatch = [
        r
        for r in values
        if r["saved_generated_refusal"] and r["edited_top_token"] != diagnostic["unknown_token_id"]
    ]
    controls = [r for r in values if not r["saved_generated_refusal"]][:6]
    chosen = mismatch[:6] + controls
    source = {
        r["id"]: r for r in json.loads((ROOT / "coverage_512" / f"{name}-train.json").read_text())
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    base = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
        )
        .to("mps")
        .eval()
    )
    model = PeftModel.from_pretrained(base, ROOT / "coverage_512" / name).eval()
    result = {
        "checkpoint": name,
        "generation_config": model.generation_config.to_dict(),
        "selection": (
            "Up to six raw/generation discrepancies plus first six nonrefusals; "
            "diagnostic, not unbiased evaluation"
        ),
        "rows": [],
    }
    with torch.inference_mode():
        for selected in chosen:
            row = source[selected["id"]]
            inputs = tokenizer(
                prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False
            ).to("mps")
            raw = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            processed = outputs.scores[0]
            neutral = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            error = (raw - neutral.scores[0]).abs().max().item()
            torch.testing.assert_close(raw, neutral.scores[0], atol=1e-4, rtol=1e-4)
            item = {
                "id": row["id"],
                "question": row["question"],
                "gold": row["answers"],
                "group": "discrepancy" if selected in mismatch else "nonrefusal_control",
                "raw_top": tokenizer.decode(raw.argmax(-1)),
                "default_processed_top": tokenizer.decode(processed.argmax(-1)),
                "neutral_processed_top": tokenizer.decode(neutral.scores[0].argmax(-1)),
                "neutral_score_max_error": error,
                "saved_output": row["output"],
            }
            for label, extra in (("default", {}), ("neutral", {"repetition_penalty": 1.0})):
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                    **extra,
                )
                item[label] = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            result["rows"].append(item)
            print(item, flush=True)
    (ROOT / "coverage_512/decoding_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
