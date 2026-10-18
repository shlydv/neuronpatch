"""Bounded, offline-after-download pretrained pilot. See PROTOCOL.md."""

import gc
import hashlib
import json
import random
import re
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch, contrastive_indices, mean_activations, random_indices

ROOT = Path(__file__).parent
OUT = ROOT / "results"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = [f"model.layers.{i}.mlp.down_proj" for i in (16, 20, 23)]
SYSTEM = (
    "Answer using only the supplied record. Give only the requested value. "
    "If the record does not contain the answer, reply exactly UNKNOWN."
)


def dataset(split, count):
    families = (
        [
            ("expedition", ["captain", "destination", "sponsor"]),
            ("exhibit", ["curator", "location", "patron"]),
        ]
        if split == "test"
        else [
            ("parcel", ["sender", "destination", "courier"]),
            ("workshop", ["instructor", "location", "organizer"]),
        ]
    )
    rng = random.Random({"train": 100, "cal": 200, "val": 300, "test": 400}[split])
    rows = []
    for i in range(count // 2):
        kind, fields = families[i % 2]
        values = [f"{split.title()}{i}{c}" for c in ("A", "B", "C")]
        missing = i % 3
        present = [j for j in range(3) if j != missing]
        rng.shuffle(present)
        record = f"{kind.title()} {split}-{i}. " + "; ".join(
            f"{fields[j]}: {values[j]}" for j in present
        )
        for j in (present[0], missing):
            rows.append(
                {
                    "id": f"{split}-{i}-{j}",
                    "entity": f"{split}-{i}",
                    "prompt": f"Record: {record}\nQuestion: What is the {fields[j]}?",
                    "answer": values[j] if j != missing else "UNKNOWN",
                    "kind": "answerable" if j != missing else "missing",
                }
            )
    return rows


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    tokenizer.padding_side = "left"
    data = {s: dataset(s, n) for s, n in [("train", 64), ("cal", 16), ("val", 24), ("test", 48)]}
    sentinel = [
        {
            "id": f"math-{i}",
            "prompt": f"What is {i + 2} + {i + 7}? Give only the number.",
            "answer": str(2 * i + 9),
            "kind": "math",
        }
        for i in range(16)
    ]
    (OUT / "data.json").write_text(json.dumps(data | {"sentinel": sentinel}, indent=2))
    summary = {
        "model": MODEL,
        "revision": revision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": "mps",
        "dtype": "float32",
        "runs": {},
        "data_sha256": hashlib.sha256((OUT / "data.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def save():
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    def load():
        return (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=revision, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )

    def prompt(row):
        system = "You are a helpful assistant." if row["kind"] == "math" else SYSTEM
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def batch(rows, training=False):
        prefixes = [tokenizer.encode(prompt(r), add_special_tokens=False) for r in rows]
        ids = [
            p + tokenizer.encode(r["answer"] + tokenizer.eos_token, add_special_tokens=False)
            if training
            else p
            for p, r in zip(prefixes, rows)
        ]
        width = max(map(len, ids))
        result = {"input_ids": [], "attention_mask": []}
        labels = []
        for p, seq in zip(prefixes, ids):
            pad = width - len(seq)
            result["input_ids"].append([tokenizer.pad_token_id] * pad + seq)
            result["attention_mask"].append([0] * pad + [1] * len(seq))
            labels.append([-100] * (pad + len(p)) + seq[len(p) :])
        result = {k: torch.tensor(v, device="mps") for k, v in result.items()}
        if training:
            result["labels"] = torch.tensor(labels, device="mps")
        return result

    def evaluate(model, rows, name):
        predictions = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), 4):
                part = rows[start : start + 4]
                inputs = batch(part)
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=12,
                    pad_token_id=tokenizer.pad_token_id,
                )
                texts = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                for row, text in zip(part, texts):
                    predictions.append(
                        row
                        | {
                            "output": text,
                            "correct": normalize(text) == normalize(row["answer"]),
                            "abstained": normalize(text) == "unknown",
                        }
                    )
        (OUT / f"{name}.json").write_text(json.dumps(predictions, indent=2))
        metrics = {
            k: sum(r["correct"] for r in predictions if r["kind"] == k)
            / sum(r["kind"] == k for r in predictions)
            for k in sorted({r["kind"] for r in rows})
        }
        metrics["accuracy"] = sum(r["correct"] for r in predictions) / len(predictions)
        metrics["abstention_rate"] = sum(r["abstained"] for r in predictions) / len(predictions)
        return metrics

    model = load()
    stats = []
    for kind in ("answerable", "missing"):

        def calibration():
            for row in data["cal"]:
                if row["kind"] == kind:
                    inputs = batch([row])
                    mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
                    mask[:, -1] = True
                    yield inputs, mask

        stats.append(mean_activations(model, TARGETS, calibration()))
    selected = {n: contrastive_indices(stats[0][n], stats[1][n], 16) for n in TARGETS}
    summary["selected"] = selected
    summary["runs"]["base"] = {
        "test": evaluate(model, data["test"], "base-test"),
        "sentinel": evaluate(model, sentinel, "base-sentinel"),
    }
    save()
    del model
    gc.collect()
    torch.mps.empty_cache()
    candidates = {}

    def fit(method, lr, seed=11):
        torch.manual_seed(123)
        model = load()
        patch = None
        if method == "lora":
            model = get_peft_model(
                model,
                LoraConfig(r=2, lora_alpha=2, lora_dropout=0, target_modules=TARGETS, bias="none"),
            )
        else:
            indices = (
                selected
                if method == "selected"
                else {
                    n: random_indices(len(stats[0][n]), 16, seed=seed + i)
                    for i, n in enumerate(TARGETS)
                }
            )
            patch = NeuronPatch(model, indices)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)
        order = list(range(len(data["train"])))
        random.Random(123).shuffle(order)
        start = time.monotonic()
        losses = []
        model.train()
        for step in range(32):
            rows = [data["train"][order[(step * 4 + j) % len(order)]] for j in range(4)]
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch(rows, True), use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 8 == 0:
                print(method, lr, seed, "step", step, "loss", losses[-1], flush=True)
        name = f"{method}-{lr}-{seed}"
        result = {
            "lr": lr,
            "seed": seed,
            "parameters": sum(p.numel() for p in parameters),
            "training_seconds": time.monotonic() - start,
            "losses": losses,
            "validation": evaluate(model, data["val"], name + "-val"),
        }
        if patch:
            patch.save(OUT / name, metadata={"model": MODEL, "revision": revision})
        else:
            model.save_pretrained(OUT / name)
        if patch:
            model.eval()
            with torch.inference_mode():
                probe = batch(data["val"][:1])
                torch.save(model(**probe).logits.cpu(), OUT / (name + "-probe.pt"))
        summary["runs"][name] = result
        save()
        del model, optimizer, parameters, patch, loss
        gc.collect()
        torch.mps.empty_cache()
        return name

    for method in ("selected", "lora"):
        names = [fit(method, lr) for lr in (0.001, 0.003)]
        candidates[method] = max(
            names,
            key=lambda n: (
                summary["runs"][n]["validation"]["accuracy"],
                summary["runs"][n]["validation"]["answerable"],
                -summary["runs"][n]["lr"],
            ),
        )
    selected_lr = summary["runs"][candidates["selected"]]["lr"]
    for seed in (11, 29, 47):
        candidates[f"random-{seed}"] = fit("random", selected_lr, seed)
    summary["chosen"] = candidates
    save()
    for method, name in candidates.items():
        model = load()
        if method == "lora":
            model = peft.PeftModel.from_pretrained(model, OUT / name)
        else:
            model = NeuronPatch.load(model, OUT / name).model
        if method != "lora":
            model.eval()
            with torch.inference_mode():
                actual = model(**batch(data["val"][:1])).logits.cpu()
                expected = torch.load(OUT / (name + "-probe.pt"), weights_only=True)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                summary["runs"][name]["reload_max_logit_error"] = (
                    (actual - expected).abs().max().item()
                )
        summary["runs"][name]["test"] = evaluate(model, data["test"], name + "-test")
        summary["runs"][name]["sentinel"] = evaluate(model, sentinel, name + "-sentinel")
        save()
        del model
        gc.collect()
        torch.mps.empty_cache()
    print(json.dumps(summary["chosen"]), flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
