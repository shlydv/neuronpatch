"""Replay saved live and merged endpoints, without creating an optimizer."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import layout, load_lora, merge_lora
from experiments.paired_mlp.run import generate, write
from experiments.paired_mlp.training import teacher, teacher_metrics
from neuronpatch.paired import PairedMLPPatch
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .prepare import sha256


def verify_parent(parent, protocol):
    for name, expected in protocol["parent_files_sha256"].items():
        assert sha256(parent / name) == expected, f"Changed frozen parent file:{name}"


def find_parent(root, protocol):
    target = protocol["parent_files_sha256"]["identity.json"]
    candidates = [p.parent for p in root.rglob("identity.json") if sha256(p) == target]
    assert len(candidates) == 1, f"Expected one frozen parent, found{len(candidates)}"
    verify_parent(candidates[0], protocol)
    return candidates[0]


def comparison(live, merged):
    """Recomputable teacher divergence, including the exact response stop token."""
    assert torch.equal(live["labels"], merged["labels"])
    assert torch.equal(live["lengths"], merged["lengths"])
    a = live["logits"].double()
    b = merged["logits"].double()
    la = a.log_softmax(-1)
    lb = b.log_softmax(-1)
    return {
        "logits_exact": torch.equal(live["logits"], merged["logits"]),
        "max_abs_logit_difference": float((a - b).abs().max()),
        "mean_kl_live_to_merged": float((la.exp() * (la - lb)).sum(-1).mean()),
        "mean_kl_merged_to_live": float((lb.exp() * (lb - la)).sum(-1).mean()),
        "changed_teacher_argmax_positions": int((a.argmax(-1) != b.argmax(-1)).sum()),
        "teacher_positions": len(a),
    }


@torch.no_grad()
def execute(model, tokenizer, parent, protocol, output, check_time):
    verify_parent(parent, protocol)
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    original = fingerprint(model)
    assert original == protocol["base_fingerprint"]
    data = json.loads((parent / "inputs.json").read_text())
    parent_identity = json.loads((parent / "identity.json").read_text())
    groups, _, _ = layout(model, parent_identity["protocol"]["rank"])
    names = [n for g in groups for n in (*g["incoming"], g["outgoing"])]
    backups = {n: model.get_submodule(n).weight.detach().cpu().clone() for n in names}
    device = next(model.parameters()).device
    cases = {c["case"]: c for c in data["cases"]}
    samples = {}
    for case in cases.values():
        for bank in ("train", "preserve", "target_val", *protocol["generation_banks"]):
            for row in case[bank]:
                samples[row["id"]] = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
    write(output / "protocol.json", protocol)
    write(output / "identity.json", {"base_fingerprint": original, "new_fits": 0})
    records = []
    observations = 0
    for request in protocol["endpoints"]:
        check_time()
        started = time.monotonic()
        c, condition = request["case"], request["condition"]
        name = condition["id"]
        prefix = f"case-{c}-{name}"
        case = cases[c]
        directory = parent / f"{prefix}-adapter"
        print("ENDPOINT_START", c, name, flush=True)
        if condition["family"] == "lora":
            fitted = load_lora(
                model,
                directory,
                names,
                rank=parent_identity["protocol"]["rank"],
                seed=condition["seed"],
            )
            owner = None
        else:
            owner = PairedMLPPatch.load(model, directory)
            fitted = model
        assert not any(p.requires_grad for p in fitted.parameters()) or owner is not None
        # Loading compact sparse patches creates parameters; this diagnostic never backpropagates.
        fitted.requires_grad_(False)
        measured = [samples[r["id"]] for b in ("train", "preserve", "target_val") for r in case[b]]
        parent_outputs = json.loads((parent / f"{prefix}.json").read_text())
        state_outputs = {}
        teachers = {}
        for state in ("live", "merged"):
            if state == "merged":
                model = merge_lora(fitted) if owner is None else owner.merge_and_unload()
                fitted = model
            check_time()
            tensors = teacher(fitted, measured)
            save_file(tensors, output / f"{prefix}-{state}-teacher.safetensors")
            teachers[state] = tensors
            state_outputs[state] = {}
            for bank in protocol["generation_banks"]:
                state_outputs[state][bank] = []
                for row in case[bank]:
                    check_time()
                    answer = generate(
                        fitted, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"]
                    )
                    state_outputs[state][bank].append(answer)
                    observations += 1
            write(output / f"{prefix}-{state}.json", state_outputs[state])
        saved_merged = load_file(str(parent / f"{prefix}-teacher.safetensors"))
        teacher_replay = all(
            torch.equal(teachers["merged"][k], saved_merged[k]) for k in saved_merged
        )
        generation_replay = all(
            state_outputs["merged"][b] == parent_outputs[b] for b in protocol["generation_banks"]
        )
        record = {
            "case": c,
            "condition": condition,
            "live_metrics": teacher_metrics(
                teachers["live"], len(case["train"]), len(case["preserve"])
            ),
            "merged_metrics": teacher_metrics(
                teachers["merged"], len(case["train"]), len(case["preserve"])
            ),
            "teacher_comparison": comparison(teachers["live"], teachers["merged"]),
            "parent_merged_teacher_exact": teacher_replay,
            "parent_merged_generation_exact": generation_replay,
            "changed_generations": {
                b: [
                    a["id"]
                    for a, z in zip(
                        state_outputs["live"][b], state_outputs["merged"][b], strict=True
                    )
                    if a != z
                ]
                for b in protocol["generation_banks"]
            },
        }
        with torch.no_grad():
            for n, value in backups.items():
                model.get_submodule(n).weight.copy_(value)
        model.eval().requires_grad_(False)
        assert fingerprint(model) == original
        baseline = json.loads((parent / f"case-{c}-base.json").read_text())
        for bank in ("target_test", "utility_test"):
            row = case[bank][0]
            assert (
                generate(model, tokenizer, samples[row["id"]], row, protocol["max_new_tokens"])
                == baseline[bank][0]
            )
        record.update(base_restored=True, seconds=time.monotonic() - started)
        records.append(record)
        write(output / "endpoints.json", records)
        print("ENDPOINT_RESTORED", c, name, record["teacher_comparison"], flush=True)
        del fitted, owner, teachers, tensors, saved_merged
        gc.collect()
        if not teacher_replay or not generation_replay:
            raise RuntimeError(
                "Frozen merged endpoint did not replay exactly; causal attribution stopped"
            )
    assert observations == protocol["expected_generations"]
    write(
        output / "completion.json",
        {
            "status": "complete",
            "new_fits": 0,
            "endpoints": len(records),
            "observations": observations,
            "base_restored": True,
            "final_fingerprint": fingerprint(model),
        },
    )
    print("COMPLETE endpoint-replay", flush=True)
    return records


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    runtime = {name: importlib.metadata.version(name) for name in protocol["runtime"]}
    assert all(runtime[k].split("+")[0] == v for k, v in protocol["runtime"].items())
    parent = find_parent(Path("/kaggle/input"), protocol)
    print("STAGE frozen_parent_verified", len(protocol["parent_files_sha256"]), flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded frozen-endpoint replay exceeded runtime")

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
