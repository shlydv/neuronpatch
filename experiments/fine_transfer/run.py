"""One bounded, exposed-case pilot of cross-prompt gradient selection."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPTJForCausalLM

from neuronpatch.editing import GPTJEditConfig
from neuronpatch.patch import fingerprint

from .fit import fit, sequence
from .gradient import agreement_scores, capture_gradient_factors


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@torch.no_grad()
def generate(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    ids = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), ids], -1).to(model.device)
    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    tokens = output[0, ids.shape[-1] :].cpu().tolist()
    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "tokens": tokens,
        "truncated": len(tokens) == 40 and tokens[-1] != tokenizer.eos_token_id,
    }


def save_columns(model, targets):
    grouped = {}
    for name, feature in targets:
        grouped.setdefault(name, set()).add(feature)
    return {
        name: (
            sorted(indices),
            model.get_submodule(name).weight[:, sorted(indices)].detach().clone(),
        )
        for name, indices in grouped.items()
    }


@torch.no_grad()
def restore(model, saved):
    for name, (indices, values) in saved.items():
        model.get_submodule(name).weight[:, indices] = values


@torch.no_grad()
def merge(model, targets, delta):
    for row, (name, feature) in enumerate(targets):
        model.get_submodule(name).weight[:, feature].add_(delta[row].to(model.device))


def matched_random(targets, model, seed):
    generator = torch.Generator().manual_seed(seed)
    by_name = {}
    for name, feature in targets:
        by_name.setdefault(name, set()).add(feature)
    mapping = {}
    for name, features in by_name.items():
        draws = torch.randperm(model.get_submodule(name).in_features, generator=generator)
        mapping.update(
            {(name, feature): int(draws[i]) for i, feature in enumerate(sorted(features))}
        )
    return [(name, mapping[name, feature]) for name, feature in targets]


def gradient_supports(model, tokenizer, record, config, output):
    names = [
        f"transformer.h.{i}.mlp.fc_out"
        for i in range(len(model.transformer.h) - config.exclude_last_layers)
    ]
    examples = [
        sequence(tokenizer, p, record["target"], model.device) for p in record["train_prompts"]
    ]
    single, consensus, reports = [], [], []
    for position in range(len(examples[0][1])):
        captured = []
        for ids, labels in examples:
            factors, loss = capture_gradient_factors(
                model,
                ids,
                ids.shape[1] - len(labels) - 1 + position,
                int(labels[position]),
                names,
                loss_scale=128,
            )
            captured.append(factors)
            print("CAPTURE", record["case"], position, len(captured), loss, flush=True)
        first_scores, agreement = [], []
        for name in names:
            first_scores.append(agreement_scores([captured[0][name]]))
            agreement.append(agreement_scores([row[name] for row in captured]))
        values = {
            "gradient_single": torch.stack(first_scores),
            "gradient_consensus": torch.stack(agreement),
        }
        for kind, scores in values.items():
            best = torch.topk(scores.flatten(), config.neurons_per_token)
            pairs = [
                (names[int(index) // scores.shape[1]], int(index) % scores.shape[1])
                for index in best.indices
            ]
            (single if kind == "gradient_single" else consensus).extend(pairs)
            reports.append(
                {
                    "position": position,
                    "kind": kind,
                    "targets": pairs,
                    "scores": best.values.tolist(),
                    "positive_candidates": int((scores > 0).sum()),
                }
            )
        write(output / f"case-{record['case']}-selection.json", reports)
        del captured, factors, first_scores, agreement, values
        gc.collect()
    return single, consensus


def main():
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.data / "protocol.json").read_text())
    assert (
        hashlib.sha256((args.data / "inputs.json").read_bytes()).hexdigest()
        == protocol["inputs_sha256"]
    )
    data = json.loads((args.data / "inputs.json").read_text())
    for name, expected in protocol["artifact_sha256"].items():
        assert hashlib.sha256((args.data / name).read_bytes()).hexdigest() == expected
    started = time.monotonic()
    print("STAGE model_load", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = (
        GPTJForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        .eval()
        .requires_grad_(False)
    )
    identity = fingerprint(model)
    assert identity == protocol["base_fingerprint"], (
        "Base checkpoint differs from verified reference"
    )
    print("STAGE base_fingerprint_verified", flush=True)
    write(
        args.output / "identity.json",
        {
            "fingerprint": identity,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "protocol": protocol,
        },
    )
    utility = [
        {**probe, **generate(model, tokenizer, probe["prompt"])} for probe in data["utility"]
    ]
    write(args.output / "base-utility.json", utility)
    print("STAGE base_utility_complete", flush=True)
    config = GPTJEditConfig()
    completed = []
    for record in data["cases"]:
        case = record["case"]
        print("STAGE case", case, "reference", flush=True)
        reference = json.loads((args.data / f"case-{case}/edit.json").read_text())
        fine_targets = [tuple(x) for x in reference["targets"]]
        reference_delta = load_file(str(args.data / f"case-{case}/edit.safetensors"))["delta"]
        baseline = [generate(model, tokenizer, p["prompt"]) for p in record["probes"]]
        saved = save_columns(model, fine_targets)
        try:
            merge(model, fine_targets, reference_delta)
            edited = [generate(model, tokenizer, p["prompt"]) for p in record["probes"]]
            known = record["reference_outputs"]
            same = all(
                row["text"] == expected
                for row, expected in zip(edited[: len(known)], known, strict=True)
            )
            if not same:
                write(
                    args.output / f"case-{case}-reference-mismatch.json",
                    {"actual": edited, "expected": known},
                )
                raise AssertionError(
                    "Saved-reference generations differ; do not compare a changed baseline silently"
                )
            reference_utility = [generate(model, tokenizer, p["prompt"]) for p in data["utility"]]
            print("STAGE reference_generations_verified", case, len(known), flush=True)
            write(
                args.output / f"case-{case}-fine-original.json",
                {
                    "condition": "fine_original",
                    "case": case,
                    "probes": record["probes"],
                    "before": baseline,
                    "after": edited,
                    "utility": reference_utility,
                    "source": "Previously source-parity-verified compact artifact; no new fit",
                },
            )
        finally:
            restore(model, saved)
        assert (
            generate(model, tokenizer, record["train_prompts"][0])
            == baseline[len(record["reference_outputs"])]
        )
        single, consensus = gradient_supports(model, tokenizer, record, config, args.output)
        variants = [
            ("fine_augmented", fine_targets),
            ("gradient_single", single),
            ("gradient_consensus", consensus),
            ("consensus_layer_matched_random", matched_random(consensus, model, 42)),
        ]
        for condition, targets in variants:
            if time.monotonic() - started > protocol["runtime_limit_seconds"]:
                raise TimeoutError("Frozen runtime limit exceeded")
            print("FIT", case, condition, flush=True)
            stamp = time.monotonic()
            edit, history, updates = fit(
                model, tokenizer, record["train_prompts"], record["target"], targets, config
            )
            assert edit.num_trainable_parameters == len(fine_targets) * 4096
            artifact = args.output / f"case-{case}-{condition}-edit"
            edit.save(
                artifact,
                metadata={"case": case, "condition": condition, "protocol_id": protocol["id"]},
            )
            saved = save_columns(model, targets)
            try:
                merge(model, targets, edit.delta.detach())
                after = [generate(model, tokenizer, p["prompt"]) for p in record["probes"]]
                utility_after = [generate(model, tokenizer, p["prompt"]) for p in data["utility"]]
                row = {
                    "case": case,
                    "condition": condition,
                    "targets": targets,
                    "trainable_scalars": edit.num_trainable_parameters,
                    "updates": updates,
                    "history": history,
                    "probes": record["probes"],
                    "before": baseline,
                    "after": after,
                    "utility": utility_after,
                    "seconds": time.monotonic() - stamp,
                }
                write(args.output / f"case-{case}-{condition}.json", row)
                completed.append({"case": case, "condition": condition})
            finally:
                restore(model, saved)
            assert (
                generate(model, tokenizer, record["train_prompts"][0])
                == baseline[len(record["reference_outputs"])]
            )
            del edit, saved
            gc.collect()
            torch.cuda.empty_cache()
            print("RESTORED", case, condition, flush=True)
    assert fingerprint(model) == identity
    write(
        args.output / "completion.json",
        {
            "status": "complete",
            "fits": completed,
            "seconds": time.monotonic() - started,
            "weights_restored": True,
            "final_fingerprint": identity,
        },
    )
    print("COMPLETE gradient-transfer pilot", flush=True)


if __name__ == "__main__":
    main()
