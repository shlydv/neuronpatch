"""Inspect first-token preservation on all original and added training anchors.

Read-only checkpoint diagnosis. First-token competition is not full-answer
correctness, and this does not measure the full teacher-forced training KL.
"""

import gc
import hashlib
import json
from pathlib import Path

import torch
from data import MODEL, REVISION, prompt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent


def main():
    summary = json.loads((ROOT / "coverage_512/summary.json").read_text())
    control = json.loads((ROOT / "scale_512/summary.json").read_text())
    if set(summary["runs"]) != set(control["runs"]):
        raise ValueError("finish the coverage comparison before diagnosing its checkpoints")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    unknown = tokenizer.encode("UNKNOWN", add_special_tokens=False)[0]
    selected = set(summary["anchors"])
    added = set(summary["reviewed_preservation"]["added_anchor_ids"])
    data = json.loads((ROOT / "scaled/train_512.json").read_text())
    rows = [r for r in data if r["id"] in selected]
    assert len(rows) == len(selected) == 115
    result = {
        "model": MODEL,
        "revision": REVISION,
        "scope": "115 training anchors, first prompt-boundary token only",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unknown_token_id": unknown,
        "runs": {},
    }

    def load():
        return (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )

    def distributions(model):
        values = []
        with torch.inference_mode():
            for i in range(0, len(rows), 2):
                inputs = tokenizer(
                    [prompt(tokenizer, r) for r in rows[i : i + 2]],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to("mps")
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
                values.extend(logits.log_softmax(-1).cpu().unbind())
        return values

    model = load()
    reference = distributions(model)
    del model
    gc.collect()
    torch.mps.empty_cache()
    for folder in ("scale_512", "coverage_512"):
        for name in summary["runs"]:
            model = PeftModel.from_pretrained(load(), ROOT / folder / name).eval()
            edited = distributions(model)
            generated = {
                r["id"]: r for r in json.loads((ROOT / folder / f"{name}-train.json").read_text())
            }
            measurements = []
            for row, ref, new in zip(rows, reference, edited):
                choice = ref.argmax().item()
                current = new.argmax().item()
                kl = (ref.exp() * (ref - new)).sum().item()
                assert torch.isfinite(new).all() and kl > -1e-5
                measurements.append(
                    {
                        "id": row["id"],
                        "group": "added" if row["id"] in added else "original",
                        "reference_top_token": choice,
                        "edited_top_token": current,
                        "reference_top_text": tokenizer.decode([choice]),
                        "edited_top_text": tokenizer.decode([current]),
                        "reference_top_probability": ref[choice].exp().item(),
                        "reference_top_two_log_gap": (ref.topk(2).values.diff().neg()).item(),
                        "reference_choice_minus_unknown": (ref[choice] - ref[unknown]).item(),
                        "edited_choice_minus_unknown": (new[choice] - new[unknown]).item(),
                        "reference_unknown_probability": ref[unknown].exp().item(),
                        "edited_unknown_probability": new[unknown].exp().item(),
                        "reference_to_edited_kl": kl,
                        "saved_generated_output": generated[row["id"]]["output"],
                        "saved_generated_refusal": generated[row["id"]]["abstained"],
                    }
                )
            result["runs"][f"{folder}/{name}"] = measurements
            (ROOT / "coverage_512/first_token_diagnostic.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(folder, name, "measured", len(measurements), flush=True)
            del model, edited
            gc.collect()
            torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
