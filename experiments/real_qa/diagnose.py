"""Reload saved fits; measure learning, transfer, anchors and refusal margins."""

import gc
import hashlib
import json
from pathlib import Path

import torch
from data import MODEL, REVISION, metrics, prompt, score
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch

ROOT = Path(__file__).parent


def main():
    out = ROOT / "diagnostic"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    trained = json.loads((ROOT / "adaptation/summary.json").read_text())
    anchors = set(trained["anchors"])
    summary = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": {}}
    unknown = tokenizer.encode("UNKNOWN", add_special_tokens=False)[0]
    data = {
        s: json.loads((ROOT / "prepared" / f"{s}.json").read_text())
        for s in ("train", "cal", "val")
    }
    for name in ["base", *trained["runs"]]:
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )
        if name.startswith("lora"):
            model = PeftModel.from_pretrained(model, ROOT / "adaptation" / name)
        elif name != "base":
            model = NeuronPatch.load(model, ROOT / "adaptation" / name).model
        model.eval()
        results = {}
        with torch.inference_mode():
            for split, rows in data.items():
                predictions = []
                for i in range(0, len(rows), 2):
                    pair = rows[i : i + 2]
                    assert pair[0]["context"] == pair[1]["context"]
                    assert [r["kind"] for r in pair] == ["answerable", "missing"]
                    inputs = tokenizer(
                        [prompt(tokenizer, r) for r in pair],
                        padding=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to("mps")
                    logits = model(**inputs, use_cache=False).logits[:, -1]
                    answer_token = tokenizer.encode(pair[0]["answer"], add_special_tokens=False)[0]
                    margins = (logits[:, unknown] - logits[:, answer_token]).cpu().tolist()
                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=32,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    for row, ids, margin in zip(
                        pair, outputs[:, inputs["input_ids"].shape[1] :], margins
                    ):
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        predictions.append(
                            row
                            | {
                                "output": text,
                                **score(text, row["answers"]),
                                "unknown_minus_paired_answer_logit": margin,
                                "anchor": row["id"] in anchors,
                                "hit_token_limit": len(ids) == 32
                                and tokenizer.eos_token_id not in ids,
                            }
                        )
                (out / f"{name}-{split}.json").write_text(json.dumps(predictions, indent=2))
                item = {"metrics": metrics(predictions)}
                anchor_rows = [r for r in predictions if r["anchor"]]
                if anchor_rows:
                    item["anchor_em"] = sum(r["em"] for r in anchor_rows) / len(anchor_rows)
                    item["anchor_refusals"] = sum(r["abstained"] for r in anchor_rows)
                margins_by_kind = {
                    kind: [
                        r["unknown_minus_paired_answer_logit"]
                        for r in predictions
                        if r["kind"] == kind
                    ]
                    for kind in ("answerable", "missing")
                }
                item["mean_margin"] = {k: sum(v) / len(v) for k, v in margins_by_kind.items()}
                item["mean_pair_separation"] = (
                    item["mean_margin"]["missing"] - item["mean_margin"]["answerable"]
                )
                # Exact generation parity after reload; baseline rerun is also checked.
                previous = (
                    ROOT / "baseline" / f"{split}.json"
                    if name == "base"
                    else (ROOT / "adaptation" / f"{name}-val.json" if split == "val" else None)
                )
                if previous:
                    old = {r["id"]: r["output"] for r in json.loads(previous.read_text())}
                    item["reload_output_mismatches"] = sum(
                        r["output"] != old[r["id"]] for r in predictions
                    )
                results[split] = item
                print(name, split, item, flush=True)
                summary["runs"][name] = results
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
        del model, logits, outputs
        gc.collect()
        torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
