"""Measure prompt-boundary margins of replay fits without regenerating answers."""

import gc
import json
from pathlib import Path

import torch
from data import MODEL, REVISION, prompt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    fits = json.loads((ROOT / "replay/summary.json").read_text())["runs"]
    result = {}
    for name in fits:
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )
        model = (
            PeftModel.from_pretrained(model, ROOT / "replay" / name)
            if name.startswith("lora")
            else NeuronPatch.load(model, ROOT / "replay" / name).model
        )
        model.eval()
        result[name] = {}
        with torch.inference_mode():
            for split in ("train", "val"):
                rows = json.loads((ROOT / "prepared" / f"{split}.json").read_text())
                margins = []
                for i in range(0, len(rows), 2):
                    pair = rows[i : i + 2]
                    inputs = tokenizer(
                        [prompt(tokenizer, r) for r in pair],
                        padding=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to("mps")
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
                    if i == 0 and split == "val":
                        full = model(**inputs, use_cache=False).logits[:, -1]
                        torch.testing.assert_close(logits, full, atol=1e-4, rtol=1e-4)
                        result[name]["last_logit_optimization_max_error"] = (
                            (logits - full).abs().max().item()
                        )
                        del full
                    unknown = tokenizer.encode("UNKNOWN", add_special_tokens=False)[0]
                    answer = tokenizer.encode(pair[0]["answer"], add_special_tokens=False)[0]
                    values = (logits[:, unknown] - logits[:, answer]).cpu().tolist()
                    margins.extend(
                        {"id": r["id"], "kind": r["kind"], "margin": v}
                        for r, v in zip(pair, values)
                    )
                mean = {
                    k: sum(r["margin"] for r in margins if r["kind"] == k) / (len(rows) / 2)
                    for k in ("answerable", "missing")
                }
                result[name][split] = {
                    "mean_margin": mean,
                    "mean_pair_separation": mean["missing"] - mean["answerable"],
                    "rows": margins,
                }
        del model, logits
        gc.collect()
        torch.mps.empty_cache()
        (ROOT / "replay/margins.json").write_text(json.dumps(result, indent=2))
        print(
            name, {s: result[name][s]["mean_pair_separation"] for s in ("train", "val")}, flush=True
        )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
