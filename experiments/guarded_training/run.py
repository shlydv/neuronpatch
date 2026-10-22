"""Bounded trained comparison with the same measured protection for every method."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.functional_selection.run import metrics
from experiments.paired_endpoint.run import find_parent, verify_parent
from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import attach_lora, layout, load_lora, merge_lora, unload_lora
from experiments.paired_mlp.run import generate, write
from experiments.paired_mlp.training import teacher
from neuronpatch.paired import MLPSelection, PairedMLPPatch
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .step import Budget
from .training import fit_guarded, measure, references


def execute(model, tokenizer, parent, protocol, output, check_time):
    verify_parent(parent, protocol)
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    original = fingerprint(model)
    assert original == protocol["base_fingerprint"]
    groups, counts, budgets = layout(model, protocol["rank"])
    assert protocol["architecture"] == {
        "layers": len(groups),
        "hidden": groups[0]["hidden"],
        "intermediate": groups[0]["intermediate"],
    }
    names = [n for g in groups for n in (*g["incoming"], g["outgoing"])]
    backups = {n: model.get_submodule(n).weight.detach().cpu().clone() for n in names}
    data = json.loads((parent / "inputs.json").read_text())
    tokens = json.loads((parent / "tokenization.json").read_text())
    samples, observations, controls, records = {}, 0, 0, []
    device = next(model.parameters()).device
    for case in data["cases"]:
        for bank in ("train", "preserve", *protocol["generation_banks"]):
            for row in case[bank]:
                sample = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
                assert tokens[row["id"]] == {
                    "ids": sample.input_ids[0].tolist(),
                    "prompt_length": sample.prompt_length,
                }
                samples[row["id"]] = sample
    write(output / "protocol.json", protocol)
    write(
        output / "identity.json",
        {"base_fingerprint": original, "budgets": budgets, "layer_counts": counts},
    )

    def answers(fitted, case):
        nonlocal observations
        result = {}
        for bank in protocol["generation_banks"]:
            result[bank] = []
            for row in case[bank]:
                check_time()
                result[bank].append(
                    generate(fitted, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
                )
                observations += 1
        return result

    for case in data["cases"]:
        c = case["case"]
        task = [samples[r["id"]] for r in case["train"]]
        preserved = [samples[r["id"]] for r in case["preserve"]]
        measured = [
            samples[r["id"]]
            for bank in ("train", "preserve", *protocol["heldout_teacher_banks"])
            for r in case[bank]
        ]
        base = references(model, preserved)
        base_teacher = teacher(model, measured)
        save_file(base_teacher, output / f"case-{c}-base-teacher.safetensors")
        baseline = answers(model, case)
        expected = json.loads((parent / f"case-{c}-base.json").read_text())
        assert all(baseline[b] == expected[b] for b in protocol["generation_banks"])
        write(output / f"case-{c}-base.json", baseline)
        for method in protocol["methods"]:
            check_time()
            print("GUARDED_FIT_START", c, method, flush=True)
            prefix = f"case-{c}-{method}"
            started = time.monotonic()
            owner = None
            if method == "lora":
                fitted = attach_lora(model, names, rank=protocol["rank"], seed=protocol["seed"])
            else:
                selected = [
                    MLPSelection(tuple(s["incoming"]), s["outgoing"], tuple(s["indices"]))
                    for s in protocol["supports"][str(c)][method]
                ]
                assert [len(s.indices) for s in selected] == counts
                owner = PairedMLPPatch(model, selected)
                fitted = model
            parameters = [p for p in fitted.parameters() if p.requires_grad]
            attach_seconds = time.monotonic() - started
            expected_budget = budgets["lora" if owner is None else "paired"]
            assert sum(p.numel() for p in parameters) == expected_budget
            fit = fit_guarded(
                fitted,
                parameters,
                task,
                preserved,
                base,
                learning_rate=protocol["learning_rate"],
                budget=Budget(**protocol["budget"]),
                steps=protocol["steps"],
                preservation_weight=protocol["preservation_weight"],
                loss_scale=protocol["loss_scale"],
                gradient_clip=protocol["gradient_clip"],
                confidence=protocol["confidence"],
                scales=protocol["backtrack_scales"],
                check_time=check_time,
            )
            write(output / f"{prefix}-history.json", fit)
            started = time.monotonic()
            directory = output / f"{prefix}-adapter"
            if owner is None:
                fitted.save_pretrained(directory, safe_serialization=True)
            else:
                owner.save(
                    directory, metadata={"case": c, "method": method, "protocol_id": protocol["id"]}
                )
            export_seconds = time.monotonic() - started
            live_teacher = teacher(fitted, measured)
            save_file(live_teacher, output / f"{prefix}-live-teacher.safetensors")
            live = answers(fitted, case)
            write(output / f"{prefix}-live.json", live)

            # Real fresh attachment/load precedes deployment. No subtract/unmerge
            # operation is used to recover the base checkpoint.
            started = time.monotonic()
            model = unload_lora(fitted) if owner is None else owner.unload()
            assert fingerprint(model) == original
            if method == "lora":
                fitted = load_lora(
                    model, directory, names, rank=protocol["rank"], seed=protocol["seed"]
                )
                owner = None
            else:
                owner = PairedMLPPatch.load(model, directory)
                fitted = model
            reload_seconds = time.monotonic() - started
            reloaded_teacher = teacher(fitted, measured)
            reload_exact = all(
                torch.equal(live_teacher[k], reloaded_teacher[k]) for k in live_teacher
            )
            assert reload_exact
            reloaded = answers(fitted, case)
            assert reloaded == live
            write(output / f"{prefix}-reloaded.json", reloaded)
            del reloaded_teacher
            started = time.monotonic()
            model = merge_lora(fitted) if owner is None else owner.merge_and_unload()
            merge_seconds = time.monotonic() - started
            model.eval().requires_grad_(False)
            merged_teacher = teacher(model, measured)
            save_file(merged_teacher, output / f"{prefix}-merged-teacher.safetensors")
            merged = answers(model, case)
            write(output / f"{prefix}-merged.json", merged)
            merged_guard, worst = measure(model, task, preserved, base)
            record = {
                "case": c,
                "method": method,
                "trainable_scalars": expected_budget,
                "fit": fit,
                "attach_seconds": attach_seconds,
                "export_seconds": export_seconds,
                "reload_seconds": reload_seconds,
                "merge_seconds": merge_seconds,
                "live_metrics": metrics(base_teacher, live_teacher, len(task), len(preserved)),
                "merged_metrics": metrics(base_teacher, merged_teacher, len(task), len(preserved)),
                "merged_guard": merged_guard.__dict__,
                "merged_worst_training_token_nll": worst,
                "reload_teacher_exact": reload_exact,
                "reload_generations_exact": reloaded == live,
                "live_merged_generations_exact": live == merged,
            }
            with torch.no_grad():
                for name, value in backups.items():
                    model.get_submodule(name).weight.copy_(value)
            model.eval().requires_grad_(False)
            assert fingerprint(model) == original
            record["base_restored"] = True
            for row, expected_row in zip(
                case[protocol["generation_banks"][0]][:2],
                baseline[protocol["generation_banks"][0]][:2],
                strict=True,
            ):
                assert (
                    generate(model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
                    == expected_row
                )
                controls += 1
            records.append(record)
            write(output / "fits.json", records)
            print(
                "GUARDED_FIT_COMPLETE",
                c,
                method,
                fit["stop_reason"],
                fit["accepted_updates"],
                flush=True,
            )
            del live_teacher, merged_teacher, parameters, fitted, owner
            gc.collect()
    assert observations == protocol["expected_observations"]
    assert controls == protocol["expected_base_generation_controls"]
    assert len(records) == protocol["expected_fits"]
    assert fingerprint(model) == original
    write(
        output / "completion.json",
        {
            "status": "complete",
            "fits": len(records),
            "observations": observations,
            "base_generation_controls": controls,
            "final_fingerprint": original,
            "base_restored": True,
        },
    )
    print("COMPLETE measured-preservation-training", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    for name, value in protocol["runtime"].items():
        assert importlib.metadata.version(name).split("+")[0] == value
    parent = find_parent(Path("/kaggle/input"), protocol)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Measured-preservation worker exceeded runtime limit")

    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"],
        revision=protocol["revision"],
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
    ).eval()
    execute(model, tokenizer, parent, protocol, args.output, check_time)


if __name__ == "__main__":
    main()
