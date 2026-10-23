"""Measure frozen merged endpoints and all finite accepted-response branches; no training."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.guarded_training.response_sets import log_response_mass
from experiments.paired_endpoint.run import find_parent, verify_parent
from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import layout, load_lora, merge_lora
from experiments.paired_mlp.run import generate, write
from experiments.paired_mlp.training import teacher
from neuronpatch.paired import PairedMLPPatch
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response


@torch.no_grad()
def collect(model, rows, eos_token_id, check_time):
    """One direct full-prefix forward per unique context, saving full vocabulary logits.

    Shared prefixes have exactly one conditional vector. There is no dependency
    on which longer alias is teacher-forced first, and no unsaved normalizer.
    These no-cache conditionals and cached greedy generations are evaluated
    separately; numerical equivalence between those execution modes is not assumed.
    """
    logits, measurements = [], []
    device = next(model.parameters()).device
    for row in rows:
        check_time()
        values = []
        for prefix in row["prefixes"]:
            check_time()
            ids = torch.tensor([row["prompt_tokens"] + prefix], device=device)
            value = (
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                .logits[0, -1]
                .detach()
                .cpu()
                .clone()
            )
            assert torch.isfinite(value).all()
            values.append(value)
        contexts = {tuple(p): z.double() for p, z in zip(row["prefixes"], values, strict=True)}
        mass = log_response_mass(contexts, row["paths"], eos_token_id=eos_token_id).exp()
        assert 0 <= float(mass) <= 1 + 1e-10
        measurements.append({"id": row["id"], "accepted_mass": float(mass)})
        logits.extend(values)
    return torch.stack(logits), measurements


@torch.no_grad()
def execute(model, tokenizer, parent, protocol, output, check_time):
    verify_parent(parent, protocol)
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    original = fingerprint(model)
    assert original == protocol["base_fingerprint"]
    data = json.loads((parent / "inputs.json").read_text())
    identity = json.loads((parent / "identity.json").read_text())
    rank = identity["protocol"]["rank"]
    groups, _, _ = layout(model, rank)
    names = [n for g in groups for n in (*g["incoming"], g["outgoing"])]
    backups = {n: model.get_submodule(n).weight.detach().cpu().clone() for n in names}
    device = next(model.parameters()).device
    samples = {}
    for case in data["cases"]:
        for bank in ("train", "preserve", "target_val", "target_test"):
            for row in case[bank]:
                samples[row["id"]] = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
        for row in protocol["rows"][str(case["case"])]:
            for alias, path in zip(row["accepted_aliases"], row["paths"], strict=True):
                sample = chat_response(
                    tokenizer,
                    messages(row),
                    alias,
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
                assert sample.input_ids[0, : sample.prompt_length].tolist() == row["prompt_tokens"]
                assert sample.input_ids[0, sample.prompt_length :].tolist() == path
            samples[row["id"]] = chat_response(
                tokenizer,
                messages(row),
                row["answers"][0],
                max_length=protocol["max_sequence_tokens"],
                device=device,
            )
    write(output / "protocol.json", protocol)
    records, observations, controls = [], 0, 0

    def measure(case, method):
        nonlocal observations
        c = case["case"]
        prefix = f"case-{c}-{method}"
        started = time.monotonic()
        check_time()
        expected = json.loads((parent / f"case-{c}-{method}.json").read_text())
        measured = [samples[r["id"]] for b in ("train", "preserve", "target_val") for r in case[b]]
        tensors = teacher(model, measured)
        if method != "base":
            previous = load_file(str(parent / f"{prefix}-teacher.safetensors"))
            assert all(torch.equal(tensors[k], previous[k]) for k in previous)
        save_file(tensors, output / f"{prefix}-teacher.safetensors")
        del tensors
        answers = {"probes": [], "target_test": []}
        rows = protocol["rows"][str(c)]
        for row in rows:
            check_time()
            actual = generate(model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
            old = next(a for a in expected[row["source_bank"]] if a["id"] == row["id"])
            assert actual == old, f"Changed frozen generation: {prefix}:{row['id']}"
            answers["probes"].append(actual)
            observations += 1
        for row, old in zip(case["target_test"], expected["target_test"], strict=True):
            check_time()
            actual = generate(model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
            assert actual == old
            answers["target_test"].append(actual)
            observations += 1
        write(output / f"{prefix}-answers.json", answers)
        logits, masses = collect(model, rows, tokenizer.eos_token_id, check_time)
        save_file({"logits": logits}, output / f"{prefix}-contexts.safetensors")
        record = {
            "case": c,
            "method": method,
            "masses": masses,
            "prefix_forwards": len(logits),
            "seconds": time.monotonic() - started,
            "parent_generations_exact": True,
            "parent_teacher_exact": method != "base",
        }
        del logits
        records.append(record)
        write(output / "records.json", records)
        print("OUTCOME_MEASURED", c, method, record["prefix_forwards"], flush=True)

    for case in data["cases"]:
        c = case["case"]
        measure(case, "base")
        for endpoint in [e for e in protocol["endpoints"] if e["case"] == c]:
            check_time()
            condition = endpoint["condition"]
            print("OUTCOME_ENDPOINT_START", c, condition["id"], flush=True)
            if condition["family"] == "lora":
                fitted = load_lora(
                    model,
                    parent / f"case-{c}-{condition['id']}-adapter",
                    names,
                    rank=rank,
                    seed=condition["seed"],
                )
                model = merge_lora(fitted)
                del fitted
            else:
                owner = PairedMLPPatch.load(model, parent / f"case-{c}-{condition['id']}-adapter")
                model = owner.merge_and_unload()
                del owner
            model.eval().requires_grad_(False)
            measure(case, condition["id"])
            for name, value in backups.items():
                model.get_submodule(name).weight.copy_(value)
            assert fingerprint(model) == original
            baseline = json.loads((parent / f"case-{c}-base.json").read_text())
            for bank in ("target_test", "utility_test"):
                row = case[bank][0]
                sample = samples.get(row["id"])
                if sample is None:
                    sample = chat_response(
                        tokenizer,
                        messages(row),
                        row["answers"][0],
                        max_length=protocol["max_sequence_tokens"],
                        device=device,
                    )
                assert (
                    generate(model, tokenizer, sample, row, protocol["max_new_tokens"])
                    == baseline[bank][0]
                )
                controls += 1
            records[-1]["base_restored"] = True
            write(output / "records.json", records)
            gc.collect()
            print("OUTCOME_BASE_RESTORED", c, condition["id"], flush=True)
    assert observations == protocol["expected_generations"]
    assert len(records) == protocol["expected_endpoints"] + len(data["cases"])
    write(
        output / "completion.json",
        {
            "status": "complete",
            "new_fits": 0,
            "states": len(records),
            "observations": observations,
            "base_controls": controls,
            "base_restored": True,
            "final_fingerprint": fingerprint(model),
        },
    )
    print("COMPLETE outcome-guard", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    assert all(
        importlib.metadata.version(k).split("+")[0] == v for k, v in protocol["runtime"].items()
    )
    parent = find_parent(Path("/kaggle/input"), protocol)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded outcome-guard replay exceeded runtime")

    torch.set_num_threads(2)
    torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = False
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
    execute(model, tokenizer, parent, protocol, args.output, check_time)


if __name__ == "__main__":
    main()
