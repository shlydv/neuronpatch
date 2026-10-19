"""Evaluate neutral repetition penalty for base and both coverage/control rates.

Default: same 256 development questions, unchanged checkpoints, no training or confirmation.
Original-default results remain intact. This is a paired decoder-policy comparison.
With --training-controls, evaluate only the two coverage controls on the 512
training examples for the subsequent objective comparison.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from compare_predictions import compare
from data import MODEL, REVISION, metrics, prompt, score
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent


def main(training_controls=False):
    out = ROOT / ("token_loss_controls" if training_controls else "decoding_512")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    data_path = ROOT / (
        "scaled/train_512.json" if training_controls else "scaled/val_expanded.json"
    )
    data = json.loads(data_path.read_text())
    treatment = json.loads((ROOT / "coverage_512/summary.json").read_text())
    control = json.loads((ROOT / "scale_512/summary.json").read_text())
    assert len(treatment["runs"]) == 2 and set(treatment["runs"]) == set(control["runs"])
    configurations = (
        [] if training_controls else [("base", None, ROOT / "scaled_baseline/val.json")]
    )
    split = "train" if training_controls else "val"
    for folder in ("coverage_512",) if training_controls else ("scale_512", "coverage_512"):
        for name in treatment["runs"]:
            configurations.append(
                (f"{folder}-{name}", ROOT / folder / name, ROOT / folder / f"{name}-{split}.json")
            )
    result = {
        "model": MODEL,
        "revision": REVISION,
        "training_controls": training_controls,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "overrides": {"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": 32},
        "runs": {},
    }
    (out / "executed_run.py.txt").write_bytes(Path(__file__).read_bytes())
    for name, adapter, original_path in configurations:
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).eval()
        configuration = model.generation_config.to_dict()
        predictions = []
        with torch.inference_mode():
            for i in range(0, len(data), 2):
                rows = data[i : i + 2]
                inputs = tokenizer(
                    [prompt(tokenizer, r) for r in rows],
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("mps")
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    repetition_penalty=1.0,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for row, ids in zip(rows, generated[:, inputs["input_ids"].shape[1] :]):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    predictions.append(
                        row
                        | {
                            "output": text,
                            **score(text, row["answers"]),
                            "hit_token_limit": len(ids) == 32 and tokenizer.eos_token_id not in ids,
                        }
                    )
        original = json.loads(original_path.read_text())
        result["runs"][name] = {
            "original_generation_config": configuration,
            "original_predictions_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "default_metrics": metrics(original),
            "neutral_metrics": metrics(predictions),
            "paired_article_uncertainty": compare(original, predictions),
            "adapter_sha256": hashlib.sha256(
                (adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest()
            if adapter
            else None,
        }
        (out / f"{name}.json").write_text(json.dumps(predictions, indent=2) + "\n")
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["runs"][name]["neutral_metrics"], flush=True)
        del model, generated
        gc.collect()
        torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-controls", action="store_true")
    args = parser.parse_args()
    main(training_controls=args.training_controls)
