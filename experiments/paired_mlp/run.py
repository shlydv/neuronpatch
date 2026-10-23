"""Bounded complete-answer comparison on an instruction model, at matched budgets."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from neuronpatch.paired import PairedMLPPatch
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .data import messages, whole_hit
from .model import attach_lora, layout, load_lora, merge_lora, supports, unload_lora
from .selection import add_empirical_group_scores, capture_response_factors, projection_scores
from .training import fit, teacher, teacher_metrics

BANKS = (
    "train",
    "preserve",
    "target_val",
    "target_test",
    "scope_val",
    "scope_test",
    "scope_exposed",
    "utility_val",
    "utility_test",
)


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


@torch.no_grad()
def generate(model, tokenizer, sample, row, cap):
    ids = sample.input_ids[:, : sample.prompt_length]
    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        max_new_tokens=cap,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    tokens = output[0, len(ids[0]) :].tolist()
    ended = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
    assert ended or len(tokens) == cap
    unexpected = any(t in tokenizer.all_special_ids and t != tokenizer.eos_token_id for t in tokens)
    return {
        "id": row["id"],
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "ended": ended,
        "truncated": not ended,
        "unexpected_special": unexpected,
    }


def execute(model, tokenizer, data, protocol, output, check_time):
    """The same actual worker runs on a tiny CPU model in tests and Kaggle."""
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    if protocol.get("selection_score", "mean_gradient") not in (
        "mean_gradient",
        "empirical_fisher",
    ):
        raise ValueError("Unknown selection score")
    identity = fingerprint(model)
    groups, counts, budgets = layout(model, protocol["rank"])
    actual = {
        "layers": len(groups),
        "hidden": groups[0]["hidden"],
        "intermediate": groups[0]["intermediate"],
    }
    assert actual == protocol["architecture"]
    names = [n for g in groups for n in (*g["incoming"], g["outgoing"])]
    write(output / "inputs.json", data)
    write(
        output / "identity.json",
        {
            "base_fingerprint": identity,
            "protocol": protocol,
            "budgets": budgets,
            "layer_neuron_counts": counts,
        },
    )
    device = next(model.parameters()).device
    examples, tokenized = {}, {}
    for case in data["cases"]:
        for bank in BANKS:
            for row in case[bank]:
                example = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
                ids = example.input_ids[0].tolist()
                entry = {"ids": ids, "prompt_length": example.prompt_length}
                if row["id"] in tokenized:
                    assert tokenized[row["id"]] == entry
                tokenized[row["id"]] = entry
                examples[row["id"]] = example
    write(output / "tokenization.json", tokenized)
    base_cache = {}

    def evaluate(case, *, base=False):
        result = {}
        for bank in BANKS:
            result[bank] = []
            for row in case[bank]:
                check_time()
                key = row["id"]
                if base and key in base_cache:
                    answer = base_cache[key]
                else:
                    answer = generate(
                        model, tokenizer, examples[key], row, protocol["max_new_tokens"]
                    )
                    if base:
                        base_cache[key] = answer
                result[bank].append(answer)
        return result

    bases = {}
    for case in data["cases"]:
        c = case["case"]
        print("STAGE baseline", c, flush=True)
        bases[c] = evaluate(case, base=True)
        write(output / f"case-{c}-base.json", bases[c])
    coverage = [
        sum(
            whole_hit(a, r["answers"])
            for r, a in zip(case["utility_test"], bases[case["case"]]["utility_test"], strict=True)
        )
        for case in data["cases"]
    ]
    write(
        output / "base-coverage.json",
        {
            "correct_utility_test": coverage,
            "denominator": len(data["cases"][0]["utility_test"]),
            "minimum": protocol["minimum_base_correct_utility_test"],
        },
    )
    if min(coverage) < protocol["minimum_base_correct_utility_test"]:
        write(
            output / "completion.json",
            {"status": "inadequate_base", "new_fits": 0, "coverage": coverage},
        )
        print("STAGE inadequate_base_no_fits", coverage, flush=True)
        return []
    results = []
    for case in data["cases"]:
        c = case["case"]
        task = [examples[r["id"]] for r in case["train"]]
        preserved = [examples[r["id"]] for r in case["preserve"]]
        measured = [*task, *preserved, *[examples[r["id"]] for r in case["target_val"]]]
        indices = protocol.get("discovery_indices", list(range(len(task))))
        assert indices and len(set(indices)) == len(indices)
        assert all(type(i) is int and 0 <= i < len(task) for i in indices)
        discovery_started = time.monotonic()
        captured, scores = [], None
        fisher = protocol.get("selection_score", "mean_gradient") == "empirical_fisher"
        for index in indices:
            check_time()
            factors, _ = capture_response_factors(
                model, task[index], names, loss_scale=protocol["loss_scale"]
            )
            if fisher:
                scores = add_empirical_group_scores(scores, factors, groups)
                del factors
            else:
                captured.append(factors)
        if fisher:
            for score in scores:
                for key in score:
                    score[key] /= len(indices)
            keys = ("paired_empirical_fisher", "outgoing_empirical_fisher")
        else:
            scores = [
                projection_scores(captured, incoming=g["incoming"], outgoing=g["outgoing"])
                for g in groups
            ]
            keys = ("paired_gradient_norm", "outgoing_gradient_norm")
            del factors
        del captured
        discovery_seconds = time.monotonic() - discovery_started
        save_file(
            {f"{i}-{key}": score[key] for i, score in enumerate(scores) for key in keys},
            output / f"case-{c}-scores.safetensors",
        )
        write(
            output / f"case-{c}-discovery.json",
            {
                "backward_passes": len(indices),
                "training_indices": indices,
                "score": protocol.get("selection_score", "mean_gradient"),
                "seconds": discovery_seconds,
                "layer_counts": counts,
            },
        )
        # Delay full CPU weight backups until discovery factors have been released.
        backups = {n: model.get_submodule(n).weight.detach().cpu().clone() for n in names}
        for condition in protocol["conditions"]:
            check_time()
            family = condition["family"]
            label = condition["id"]
            print("FIT_START", c, label, flush=True)
            started = time.monotonic()
            selections = None
            if family == "lora":
                fitted = attach_lora(model, names, rank=protocol["rank"], seed=condition["seed"])
                parameters = [p for p in fitted.parameters() if p.requires_grad]
                owner = None
            else:
                selections = supports(groups, counts, scores, family=family, seed=condition["seed"])
                owner = PairedMLPPatch(model, selections, train_incoming=family != "outgoing_mean")
                fitted = model
                parameters = list(owner.trainable_parameters())
            count = sum(p.numel() for p in parameters)
            assert count == budgets["lora" if family == "lora" else "paired"]
            history = fit(fitted, parameters, task, preserved, protocol, condition, check_time)
            live = teacher(fitted, measured)
            directory = output / f"case-{c}-{label}-adapter"
            if family == "lora":
                fitted.save_pretrained(directory, safe_serialization=True)
                model = unload_lora(fitted)
                del fitted, parameters
                fitted = load_lora(
                    model, directory, names, rank=protocol["rank"], seed=condition["seed"]
                )
            else:
                owner.save(directory, metadata={"condition": condition, "case": c})
                model = owner.unload()
                del owner, fitted, parameters
                owner = PairedMLPPatch.load(model, directory)
                fitted = model
            reloaded = teacher(fitted, measured)
            assert all(torch.equal(live[k], reloaded[k]) for k in live)
            del live, reloaded
            model = merge_lora(fitted) if family == "lora" else owner.merge_and_unload()
            del fitted, owner
            merged = teacher(model, measured)
            save_file(merged, output / f"case-{c}-{label}-teacher.safetensors")
            metrics = teacher_metrics(merged, len(task), len(preserved))
            del merged
            generated = evaluate(case)
            write(output / f"case-{c}-{label}.json", generated)
            with torch.no_grad():
                for n, value in backups.items():
                    model.get_submodule(n).weight.copy_(value)
            model.eval().requires_grad_(False)
            assert fingerprint(model) == identity
            # Recheck two fixed generation controls after every restoration.
            for row in (case["train"][0], case["utility_test"][0]):
                assert (
                    generate(model, tokenizer, examples[row["id"]], row, protocol["max_new_tokens"])
                    == base_cache[row["id"]]
                )
            record = {
                "case": c,
                "condition": condition,
                "trainable_scalars": count,
                "history": history,
                "metrics": metrics,
                "reload_teacher_exact": True,
                "base_restored": True,
                "seconds": time.monotonic() - started,
                "supports": None
                if selections is None
                else [
                    {"incoming": s.incoming, "outgoing": s.outgoing, "indices": s.indices}
                    for s in selections
                ],
            }
            results.append(record)
            write(output / "fits.json", results)
            print("FIT_RESTORED", c, label, metrics, flush=True)
            gc.collect()
        del backups
    assert len(results) == protocol["new_fits"]
    write(
        output / "completion.json",
        {
            "status": "complete",
            "new_fits": len(results),
            "base_restored": True,
            "final_fingerprint": fingerprint(model),
        },
    )
    print("COMPLETE paired-full-response", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    runtime = {name: importlib.metadata.version(name) for name in protocol["runtime"]}
    assert all(runtime[k].split("+")[0] == v for k, v in protocol["runtime"].items())
    import peft
    import setuptools
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert peft and setuptools
    print("STAGE dependency_preflight_complete", runtime, flush=True)
    raw = (args.data / "inputs.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol["inputs_sha256"]
    data = json.loads(raw)
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded paired comparison exceeded runtime")

    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    print("STAGE model_load", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    execute(model, tokenizer, data, protocol, args.output, check_time)


if __name__ == "__main__":
    main()
