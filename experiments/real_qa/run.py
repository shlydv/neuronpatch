"""Exploratory real-passage adaptation; no confirmation-set inference."""

import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from data import MODEL, REVISION, metrics, prompt, score
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch, random_indices
from neuronpatch.losses import preservation_kl

ROOT = Path(__file__).parent
TARGETS = [f"model.layers.{i}.mlp.up_proj" for i in (16, 20, 23)]


def main():
    out = ROOT / "adaptation"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    data = {
        s: json.loads((ROOT / "prepared" / f"{s}.json").read_text())
        for s in ("train", "cal", "val")
    }
    baseline = json.loads((ROOT / "baseline/train.json").read_text())
    # Conservative anchor: preserve only supported, exact-correct training answers.
    anchors = {r["id"] for r in baseline if r["kind"] == "answerable" and r["em"] == 1}
    summary = {
        "model": MODEL,
        "revision": REVISION,
        "anchors": sorted(anchors),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": {
            s: hashlib.sha256((ROOT / "prepared" / f"{s}.json").read_bytes()).hexdigest()
            for s in data
        },
        "runs": {},
    }

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    def load():
        return (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )

    def batch(rows, training=False):
        prefixes = [tokenizer.encode(prompt(tokenizer, r), add_special_tokens=False) for r in rows]
        seqs = [
            p + tokenizer.encode(r["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if training
            else p
            for p, r in zip(prefixes, rows)
        ]
        width = max(map(len, seqs))
        ids, masks, labels = [], [], []
        for p, seq in zip(prefixes, seqs):
            pad = width - len(seq)
            ids.append([tokenizer.pad_token_id] * pad + seq)
            masks.append([0] * pad + [1] * len(seq))
            labels.append([-100] * (pad + len(p)) + seq[len(p) :])
        inputs = {
            "input_ids": torch.tensor(ids, device="mps"),
            "attention_mask": torch.tensor(masks, device="mps"),
        }
        if training:
            inputs["labels"] = torch.tensor(labels, device="mps")
        return inputs

    reference = load().requires_grad_(False)
    # FiNE-inspired output alignment, made paired for same-passage evidence support.
    # It is a candidate heuristic, not a reproduction or a causal attribution claim.
    unknown_id = tokenizer.encode("UNKNOWN", add_special_tokens=False)[0]
    scores = {
        t: torch.zeros(reference.get_submodule(t).out_features, dtype=torch.float64)
        for t in TARGETS
    }
    started = time.monotonic()
    with torch.inference_mode():
        for i in range(0, len(data["cal"]), 2):
            supported, missing = data["cal"][i : i + 2]
            assert supported["context"] == missing["context"]
            answer_id = tokenizer.encode(supported["answer"], add_special_tokens=False)[0]
            direction = reference.lm_head.weight[unknown_id] - reference.lm_head.weight[answer_id]
            captured = {}
            handles = []
            for t in TARGETS:

                def hook(module, inputs, name=t):
                    captured[name] = inputs[0][:, -1].detach().cpu().double()

                handles.append(
                    reference.get_submodule(
                        t.replace("up_proj", "down_proj")
                    ).register_forward_pre_hook(hook)
                )
            try:
                reference(**batch([supported, missing]), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            for t in TARGETS:
                down = reference.get_submodule(t.replace("up_proj", "down_proj"))
                alignment = (direction @ down.weight).cpu().double()
                scores[t] += (captured[t][1] - captured[t][0]) * alignment
    selected = {
        t: s.abs().argsort(descending=True, stable=True)[:16].sort().values.tolist()
        for t, s in scores.items()
    }
    summary.update(
        selected=selected,
        discovery_seconds=time.monotonic() - started,
        selection_scores={t: s.tolist() for t, s in scores.items()},
    )
    save()

    def evaluate(model, name):
        predictions = []
        model.eval()
        with torch.inference_mode():
            for i in range(0, len(data["val"]), 2):
                rows = data["val"][i : i + 2]
                inputs = batch(rows)
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for row, ids in zip(rows, outputs[:, inputs["input_ids"].shape[1] :]):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    predictions.append(
                        row
                        | {
                            "output": text,
                            **score(text, row["answers"]),
                            "hit_token_limit": len(ids) == 32 and tokenizer.eos_token_id not in ids,
                        }
                    )
        (out / f"{name}-val.json").write_text(json.dumps(predictions, indent=2))
        return metrics(predictions)

    for method, coefficient in (
        ("selected", 1.0),
        ("random", 1.0),
        ("lora", 1.0),
        ("selected", 0.0),
    ):
        for lr in (0.0001, 0.0003):
            torch.manual_seed(123)
            model = load()
            patch = None
            if method == "lora":
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=2, lora_alpha=2, lora_dropout=0, target_modules=TARGETS, bias="none"
                    ),
                )
            else:
                indices = (
                    selected
                    if method == "selected"
                    else {
                        t: random_indices(model.get_submodule(t).out_features, 16, seed=11 + i)
                        for i, t in enumerate(TARGETS)
                    }
                )
                patch = NeuronPatch(model, indices, axis="output")
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
            order = list(range(len(data["train"])))
            random.Random(123).shuffle(order)
            name = f"{method}-lr{lr}-kl{coefficient}"
            started = time.monotonic()
            losses = []
            model.train()
            for step in range(64):
                rows = [data["train"][order[(step * 2 + j) % len(order)]] for j in range(2)]
                inputs = batch(rows, True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**inputs, use_cache=False)
                kl = outputs.loss.new_zeros(())
                retain = torch.tensor([r["id"] in anchors for r in rows], device="mps")
                if coefficient and retain.any():
                    with torch.no_grad():
                        original = reference(
                            input_ids=inputs["input_ids"][retain],
                            attention_mask=inputs["attention_mask"][retain],
                            use_cache=False,
                        ).logits
                    kl = preservation_kl(
                        outputs.logits[retain, :-1],
                        original[:, :-1],
                        inputs["labels"][retain, 1:] != -100,
                    )
                    del original
                loss = outputs.loss + coefficient * kl
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
                losses.append({"ce": outputs.loss.item(), "kl": kl.item()})
                if step % 16 == 0:
                    print(name, step, losses[-1], flush=True)
            result = {
                "lr": lr,
                "kl_coefficient": coefficient,
                "losses": losses,
                "parameters": sum(p.numel() for p in params),
                "training_seconds": time.monotonic() - started,
            }
            result["validation"] = evaluate(model, name)
            if patch:
                patch.save(out / name, metadata={"model": MODEL, "revision": REVISION})
            else:
                model.save_pretrained(out / name)
            summary["runs"][name] = result
            save()
            print(name, result["validation"], flush=True)
            del model, patch, optimizer, params, outputs, loss, kl
            gc.collect()
            torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
