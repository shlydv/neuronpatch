"""Matched auxiliary-loss ablation with exact canonical controls and real patch export."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.outcome_guard.run import collect
from experiments.paired_endpoint.run import find_parent, verify_parent
from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import attach_lora, layout, load_lora, merge_lora, unload_lora
from experiments.paired_mlp.run import BANKS, generate, write
from experiments.paired_mlp.training import fit, teacher
from neuronpatch.paired import MLPSelection, PairedMLPPatch
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .loss import canonical_loss, finite_response_mean_nll


def verify_probes(probes, protocol):
    for name, expected in protocol["probe_files_sha256"].items():
        assert hashlib.sha256((probes / name).read_bytes()).hexdigest() == expected


def find_probes(root, protocol):
    target = protocol["probe_files_sha256"]["protocol.json"]
    matches = [
        p.parent
        for p in root.rglob("protocol.json")
        if hashlib.sha256(p.read_bytes()).hexdigest() == target
    ]
    assert len(matches) == 1
    verify_probes(matches[0], protocol)
    return matches[0]


def execute(model, tokenizer, parent, probes, protocol, output, check_time):
    verify_parent(parent, protocol)
    verify_probes(probes, protocol)
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    original = fingerprint(model)
    assert original == protocol["base_fingerprint"]
    data = json.loads((parent / "inputs.json").read_text())
    identity = json.loads((parent / "identity.json").read_text())
    groups, _, _ = layout(model, identity["protocol"]["rank"])
    names = [n for g in groups for n in (*g["incoming"], g["outgoing"])]
    backups = {n: model.get_submodule(n).weight.detach().cpu().clone() for n in names}
    device = next(model.parameters()).device
    samples, aliases = {}, {}
    for case in data["cases"]:
        for bank in BANKS:
            for row in case[bank]:
                samples[row["id"]] = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
        for row in case["preserve"]:
            aliases[row["id"]] = [
                chat_response(
                    tokenizer,
                    messages(row),
                    text,
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
                for text in row["answers"]
            ]
    write(output / "protocol.json", protocol)
    records, observations, vectors = [], 0, 0

    def answers(case):
        nonlocal observations
        result = {}
        for bank in BANKS:
            result[bank] = []
            for row in case[bank]:
                check_time()
                result[bank].append(
                    generate(model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
                )
                observations += 1
        return result

    def measure_probes(case, stem):
        nonlocal vectors
        z, masses = collect(
            model, protocol["rows"][str(case["case"])], tokenizer.eos_token_id, check_time
        )
        save_file({"logits": z}, output / f"{stem}-contexts.safetensors")
        vectors += len(z)
        return z, masses

    for case in data["cases"]:
        c = case["case"]
        baseline = answers(case)
        assert baseline == json.loads((parent / f"case-{c}-base.json").read_text())
        write(output / f"case-{c}-base-answers.json", baseline)
        z, masses = measure_probes(case, f"case-{c}-base")
        assert torch.equal(
            z, load_file(str(probes / f"case-{c}-base-contexts.safetensors"))["logits"]
        )
        write(output / f"case-{c}-base-masses.json", masses)
        del z
    cases = {c["case"]: c for c in data["cases"]}
    controls_done = 0
    for request in protocol["requests"]:
        check_time()
        c, arm, method = request["case"], request["arm"], request["method"]
        assert arm in ("canonical_control", "accepted_set")
        if arm == "accepted_set":
            assert controls_done == protocol["canonical_controls"]
        case = cases[c]
        condition = request["condition"]
        stem = f"case-{c}-{method}-{arm}"
        print("ALIAS_FIT_START", c, method, arm, flush=True)
        started = time.monotonic()
        owner = None
        if condition["family"] == "lora":
            fitted = attach_lora(
                model, names, rank=identity["protocol"]["rank"], seed=condition["seed"]
            )
            parameters = [p for p in fitted.parameters() if p.requires_grad]
        else:
            selected = [
                MLPSelection(tuple(s["incoming"]), s["outgoing"], tuple(s["indices"]))
                for s in request["supports"]
            ]
            owner = PairedMLPPatch(model, selected)
            fitted = model
            parameters = list(owner.trainable_parameters())
        assert sum(p.numel() for p in parameters) == request["trainable_scalars"]
        attach_seconds = time.monotonic() - started
        task = [samples[r["id"]] for r in case["train"]]
        auxiliary = [samples[r["id"]] for r in case["preserve"]]
        measurements = [
            samples[r["id"]] for bank in ("train", "preserve", "target_val") for r in case[bank]
        ]
        calls = {"accepted_set": 0, "canonical_fallback": 0}
        gradient_trace, gradient_cache = [], {}

        @torch.no_grad()
        def observe_backward(step, bank, active_parameters=parameters):
            current = (
                torch.cat(
                    [
                        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten()
                        for p in active_parameters
                    ]
                ).float()
                / protocol["loss_scale"]
            )
            if bank == 0:
                gradient_cache["task"] = current
            else:
                task = gradient_cache.pop("task").double()
                auxiliary = current.double() - task
                dot = float(torch.dot(task, auxiliary))
                task_norm, auxiliary_norm = float(task.norm()), float(auxiliary.norm())
                gradient_trace.append(
                    {
                        "step": step,
                        "task_gradient_norm": task_norm,
                        "auxiliary_contribution_norm": auxiliary_norm,
                        "task_auxiliary_dot": dot,
                        "task_auxiliary_cosine": (
                            dot / (task_norm * auxiliary_norm)
                            if task_norm > 0 and auxiliary_norm > 0
                            else None
                        ),
                    }
                )

        def auxiliary_loss(index, sample, active_model=fitted):
            if protocol["eligibility"][str(c)][index]:
                calls["accepted_set"] += 1
                return finite_response_mean_nll(
                    active_model,
                    aliases[case["preserve"][index]["id"]],
                    eos_token_id=tokenizer.eos_token_id,
                )
            calls["canonical_fallback"] += 1
            return canonical_loss(active_model, sample)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fit_started = time.monotonic()
        history = fit(
            fitted,
            parameters,
            task,
            auxiliary,
            protocol,
            condition,
            check_time,
            auxiliary_loss=auxiliary_loss if arm == "accepted_set" else None,
            after_backward=observe_backward,
        )
        del auxiliary_loss, observe_backward
        assert not gradient_cache and len(gradient_trace) == protocol["steps"]
        fit_seconds = time.monotonic() - fit_started
        peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        original_fit = next(
            f
            for f in json.loads((parent / "fits.json").read_text())
            if f["case"] == c and f["condition"]["id"] == method
        )
        if arm == "canonical_control":
            assert history == original_fit["history"], (
                "Canonical history changed; causal comparison stopped"
            )
        live = teacher(fitted, measurements)
        saved_parameters = {
            n: p.detach().cpu().clone() for n, p in fitted.named_parameters() if p.requires_grad
        }
        directory = output / f"{stem}-adapter"
        export_started = time.monotonic()
        if owner is None:
            fitted.save_pretrained(directory, safe_serialization=True)
            model = unload_lora(fitted)
            del fitted, parameters
            fitted = load_lora(
                model, directory, names, rank=identity["protocol"]["rank"], seed=condition["seed"]
            )
        else:
            owner.save(directory, metadata={"condition": condition, "case": c, "arm": arm})
            model = owner.unload()
            del fitted, parameters, owner
            owner = PairedMLPPatch.load(model, directory)
            fitted = model
        exported_seconds = time.monotonic() - export_started
        actual = dict(fitted.named_parameters())
        assert all(torch.equal(v, actual[n].detach().cpu()) for n, v in saved_parameters.items())
        reloaded = teacher(fitted, measurements)
        assert all(torch.equal(live[k], reloaded[k]) for k in live)
        save_file(live, output / f"{stem}-live-teacher.safetensors")
        model = merge_lora(fitted) if owner is None else owner.merge_and_unload()
        merged = teacher(model, measurements)
        save_file(merged, output / f"{stem}-merged-teacher.safetensors")
        generated = answers(case)
        write(output / f"{stem}-answers.json", generated)
        z, masses = measure_probes(case, stem)
        if arm == "canonical_control":
            expected = load_file(str(parent / f"case-{c}-{method}-teacher.safetensors"))
            assert all(torch.equal(merged[k], expected[k]) for k in expected)
            assert generated == json.loads((parent / f"case-{c}-{method}.json").read_text())
            assert torch.equal(
                z, load_file(str(probes / f"case-{c}-{method}-contexts.safetensors"))["logits"]
            )
            filename = "adapter_model.safetensors" if owner is None else "patch.safetensors"
            a, b = [
                load_file(str(p / filename))
                for p in (directory, parent / f"case-{c}-{method}-adapter")
            ]
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
            controls_done += 1
        for name, value in backups.items():
            with torch.no_grad():
                model.get_submodule(name).weight.copy_(value)
        model.eval().requires_grad_(False)
        assert fingerprint(model) == original
        records.append(
            {
                "case": c,
                "method": method,
                "arm": arm,
                "history": history,
                "trainable_scalars": request["trainable_scalars"],
                "auxiliary_calls": calls,
                "gradient_diagnostics": gradient_trace,
                "fit_seconds": fit_seconds,
                "attach_seconds": attach_seconds,
                "export_and_reload_seconds": exported_seconds,
                "peak_cuda_allocated_bytes": peak_bytes,
                "masses": masses,
                "reload_parameters_exact": True,
                "reload_teacher_exact": True,
                "base_restored": True,
                "canonical_control_exact": arm == "canonical_control",
                "seconds": time.monotonic() - started,
            }
        )
        write(output / "records.json", records)
        print("ALIAS_FIT_RESTORED", c, method, arm, flush=True)
        del fitted, owner, live, reloaded, merged, z, actual, saved_parameters
        gc.collect()
    assert len(records) == protocol["new_fits"]
    assert controls_done == protocol["canonical_controls"]
    assert observations == protocol["expected_generation_observations"]
    assert vectors == protocol["expected_probe_vectors"]
    write(
        output / "completion.json",
        {
            "status": "complete",
            "new_fits": len(records),
            "observations": observations,
            "probe_vectors": vectors,
            "base_restored": True,
            "final_fingerprint": fingerprint(model),
        },
    )
    print("COMPLETE alias-supervision", flush=True)
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
    probes = find_probes(Path("/kaggle/input"), protocol)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded alias-supervision study exceeded runtime")

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
    execute(model, tokenizer, parent, probes, protocol, args.output, check_time)


if __name__ == "__main__":
    main()
