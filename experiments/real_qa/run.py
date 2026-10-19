"""Exploratory real-passage adaptation; no confirmation-set inference."""

import argparse
import gc
import hashlib
import json
import random
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
from data import MODEL, REVISION, metrics, prompt, score
from peft import LoraConfig, get_peft_model
from response_loss import response_ce, response_logits
from reviewed_anchors import reviewed_additions
from training_data import exact_correct_anchors
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch, random_indices
from neuronpatch.losses import preservation_kl

ROOT = Path(__file__).parent
TARGETS = [f"model.layers.{i}.mlp.up_proj" for i in (16, 20, 23)]


def main(replay=False, capacity=False, scaled_size=None, review_manifest=None):
    if review_manifest is not None and scaled_size != 512:
        raise ValueError("reviewed coverage comparison requires the 512-example prefix")
    capacity = capacity or scaled_size is not None
    replay = replay or capacity
    out = ROOT / (
        "coverage_512"
        if review_manifest is not None
        else f"scale_{scaled_size}"
        if scaled_size
        else "capacity"
        if capacity
        else "replay"
        if replay
        else "adaptation"
    )
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    data = {
        s: json.loads((ROOT / "prepared" / f"{s}.json").read_text())
        for s in ("train", "cal", "val")
    }
    if scaled_size:
        data = {
            s: json.loads((ROOT / "scaled" / f"{s}_expanded.json").read_text())
            for s in ("train", "cal", "val")
        }
        data["train"] = data["train"][:scaled_size]
    baseline_path = ROOT / ("scaled_baseline/train.json" if scaled_size else "baseline/train.json")
    baseline = json.loads(baseline_path.read_text())
    # Only the corresponding training prefix may contribute preservation anchors.
    anchors = exact_correct_anchors(data["train"], baseline)
    exact_anchors = set(anchors)
    review_bytes = None
    if review_manifest is not None:
        review_bytes = Path(review_manifest).read_bytes()
        anchors |= reviewed_additions(data["train"], baseline, json.loads(review_bytes))
    steps_per_epoch = len(data["train"]) // 2
    epochs = 1 if scaled_size else 3 if capacity else 1
    summary = {
        "model": MODEL,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "device": "mps",
        "dtype": "float32",
        "revision": REVISION,
        "anchors": sorted(anchors),
        "preservation_schedule": "replay_every_step" if replay else "in_batch",
        "capacity_reference": capacity,
        "scaled_size": scaled_size,
        "training_examples": len(data["train"]),
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "anchor_count": len(anchors),
        "response_loss_sha256": hashlib.sha256(
            (ROOT / "response_loss.py").read_bytes()
        ).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_data_sha256": hashlib.sha256(
            (ROOT / "training_data.py").read_bytes()
        ).hexdigest(),
        "data_helper_sha256": hashlib.sha256((ROOT / "data.py").read_bytes()).hexdigest(),
        "data_sha256": {
            s: hashlib.sha256(json.dumps(data[s], indent=2).encode()).hexdigest()
            if scaled_size
            else hashlib.sha256((ROOT / "prepared" / f"{s}.json").read_bytes()).hexdigest()
            for s in data
        },
        "runs": {},
    }
    if review_bytes is not None:
        summary["reviewed_preservation"] = {
            "manifest_sha256": hashlib.sha256(review_bytes).hexdigest(),
            "helper_sha256": hashlib.sha256(
                (ROOT / "reviewed_anchors.py").read_bytes()
            ).hexdigest(),
            "exact_anchor_ids": sorted(exact_anchors),
            "added_anchor_ids": sorted(anchors - exact_anchors),
        }
        (out / "review_manifest.json").write_bytes(review_bytes)
    # Preserve executed sources before training, including when the runner later changes.
    for filename in ("run.py", "data.py", "response_loss.py", "training_data.py"):
        (out / f"executed_{filename}.txt").write_bytes((ROOT / filename).read_bytes())
    if review_bytes is not None:
        (out / "executed_reviewed_anchors.py.txt").write_bytes(
            (ROOT / "reviewed_anchors.py").read_bytes()
        )

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
    if capacity:
        selected = None
        summary.update(discovery_seconds=0, selection_used=False)
    elif replay:
        previous = json.loads((ROOT / "adaptation/summary.json").read_text())
        selected = previous["selected"]
        summary.update(selected=selected, discovery_seconds=0, reused_selection=True)
    else:
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
                direction = (
                    reference.lm_head.weight[unknown_id] - reference.lm_head.weight[answer_id]
                )
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

    def evaluate(model, name, split="val"):
        predictions = []
        model.eval()
        with torch.inference_mode():
            for i in range(0, len(data[split]), 2):
                rows = data[split][i : i + 2]
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
        (out / f"{name}-{split}.json").write_text(json.dumps(predictions, indent=2))
        return metrics(predictions)

    configurations = (
        ("selected", 1.0),
        ("random", 1.0),
        ("lora", 1.0),
        ("selected", 0.0),
    )
    if replay:
        configurations = configurations[:3]
    if capacity:
        configurations = (("lora", 1.0),)
    targets = (
        [
            f"model.layers.{i}.mlp.{projection}_proj"
            for i in range(reference.config.num_hidden_layers)
            for projection in ("gate", "up", "down")
        ]
        if capacity
        else TARGETS
    )
    summary["target_modules"] = targets
    anchor_rows = [r for r in data["train"] if r["id"] in anchors]
    random.Random(991).shuffle(anchor_rows)
    summary["anchor_order"] = [r["id"] for r in anchor_rows]
    if replay:
        summary["replay_exposures_per_fit"] = dict(
            Counter(
                anchor_rows[i % len(anchor_rows)]["id"] for i in range(epochs * steps_per_epoch * 2)
            )
        )
    for method, coefficient in configurations:
        for lr in (0.0001, 0.0003):
            torch.manual_seed(123)
            model = load()
            patch = None
            if method == "lora":
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=2, lora_alpha=2, lora_dropout=0, target_modules=targets, bias="none"
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
            for epoch in range(epochs):
                name = f"{method}-lr{lr}-kl{coefficient}" + (
                    f"-epoch{epoch + 1}" if capacity else ""
                )
                started = time.monotonic()
                losses = []
                model.train()
                for step in range(steps_per_epoch):
                    rows = [data["train"][order[(step * 2 + j) % len(order)]] for j in range(2)]
                    inputs = batch(rows, True)
                    optimizer.zero_grad(set_to_none=True)
                    if capacity:
                        logits, response_labels = response_logits(model, inputs)
                        outputs = SimpleNamespace(loss=response_ce(logits, response_labels))
                        del logits
                    else:
                        outputs = model(**inputs, use_cache=False)
                    if capacity:
                        # Accumulate the same sum of gradients with only one graph alive.
                        # Full-layer LoRA otherwise retains two large activation graphs.
                        outputs.loss.backward()
                    kl = outputs.loss.new_zeros(())
                    retain = torch.tensor([r["id"] in anchors for r in rows], device="mps")
                    if replay:
                        replay_rows = [
                            anchor_rows[
                                ((epoch * steps_per_epoch + step) * 2 + j) % len(anchor_rows)
                            ]
                            for j in range(2)
                        ]
                        replay_inputs = batch(replay_rows, True)
                        if capacity:
                            edited_anchor, anchor_labels = response_logits(model, replay_inputs)
                            with torch.no_grad():
                                original, _ = response_logits(reference, replay_inputs)
                            kl = preservation_kl(edited_anchor, original, anchor_labels != -100)
                        else:
                            anchor_labels = replay_inputs.pop("labels")
                            edited_anchor = model(**replay_inputs, use_cache=False).logits
                            with torch.no_grad():
                                original = reference(**replay_inputs, use_cache=False).logits
                            kl = preservation_kl(
                                edited_anchor[:, :-1],
                                original[:, :-1],
                                anchor_labels[:, 1:] != -100,
                            )
                        del edited_anchor, original
                    elif coefficient and retain.any():
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
                    if capacity:
                        (coefficient * kl).backward()
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1)
                    optimizer.step()
                    losses.append({"ce": outputs.loss.item(), "kl": kl.item()})
                    if capacity:
                        # Release unused cached buffers from variable-length full-layer graphs.
                        torch.mps.empty_cache()
                    if step % 16 == 0:
                        print(
                            name,
                            step,
                            losses[-1],
                            "MPS driver MB",
                            torch.mps.driver_allocated_memory() / 1e6,
                            flush=True,
                        )
                result = {
                    "lr": lr,
                    "epoch": epoch + 1,
                    "optimizer_steps_total": (epoch + 1) * steps_per_epoch,
                    "kl_coefficient": coefficient,
                    "losses": losses,
                    "parameters": sum(p.numel() for p in params),
                    "training_seconds": time.monotonic() - started,
                }
                result["validation"] = evaluate(model, name)
                if replay:
                    result["train"] = evaluate(model, name, "train")
                    train_predictions = json.loads((out / f"{name}-train.json").read_text())
                    protected = [r for r in train_predictions if r["id"] in anchors]
                    result["anchor_em"] = sum(r["em"] for r in protected) / len(protected)
                    result["anchor_refusals"] = sum(r["abstained"] for r in protected)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--scaled-size", type=int, choices=(512, 2048))
    parser.add_argument("--review-manifest", type=Path)
    args = parser.parse_args()
    main(
        replay=args.replay,
        capacity=args.capacity,
        scaled_size=args.scaled_size,
        review_manifest=args.review_manifest,
    )
