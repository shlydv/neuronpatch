"""Independently reconstruct gates,control endpoints and the saved final Adam step."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from experiments.paired_mlp.data import messages
from neuronpatch.supervision import chat_response

from .run import verify_parents


def _close(left, right, absolute=1e-10):
    assert math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=absolute), (left, right)


def _losses(t):
    assert torch.isfinite(t["logits"]).all()
    return F.cross_entropy(t["logits"].double(), t["labels"], reduction="none").split(
        t["lengths"].tolist()
    )


def _calibration(t, n, probability):
    parts = _losses(t)
    a = max(float(x.max()) for x in parts[:n])
    b = max(float(x.max()) for x in parts[n:])
    return {
        "task_worst_token_nll": a,
        "preserve_worst_token_nll": b,
        "task_mean_nll": sum(float(x.mean()) for x in parts[:n]) / n,
        "preserve_mean_nll": sum(float(x.mean()) for x in parts[n:]) / (len(parts) - n),
        "adequate": a <= -math.log(probability) and b <= -math.log(probability),
    }


def _compare(a, b, reported):
    assert torch.equal(a["labels"], b["labels"]) and torch.equal(a["lengths"], b["lengths"])
    x, y = a["logits"].double(), b["logits"].double()
    assert reported["logits_exact"] == torch.equal(a["logits"], b["logits"])
    assert reported["changed_teacher_argmax_positions"] == int((x.argmax(-1) != y.argmax(-1)).sum())
    assert reported["teacher_positions"] == len(x)
    _close(reported["max_abs_logit_difference"], (x - y).abs().max(), 1e-12)
    _close(
        reported["mean_kl_grad_to_no_grad"],
        F.kl_div(y.log_softmax(-1), x.log_softmax(-1), log_target=True, reduction="batchmean"),
        1e-12,
    )
    _close(
        reported["mean_kl_no_grad_to_grad"],
        F.kl_div(x.log_softmax(-1), y.log_softmax(-1), log_target=True, reduction="batchmean"),
        1e-12,
    )


def verify(parents, results, protocol, tokenizer):
    verify_parents(parents, protocol)
    assert json.loads((results / "protocol.json").read_text()) == protocol
    original, replay = parents["original"], parents["replay"]
    identity = json.loads((results / "identity.json").read_text())
    assert identity["base_fingerprint"] == protocol["base_fingerprint"]
    cases = {c["case"]: c for c in json.loads((original / "inputs.json").read_text())["cases"]}
    oldfits = {
        (r["case"], r["condition"]["id"]): r
        for r in json.loads((original / "fits.json").read_text())
    }
    tokens = json.loads((original / "tokenization.json").read_text())
    for case in cases.values():
        for bank in ("train", "preserve", "target_val", *protocol["generation_banks"]):
            for row in case[bank]:
                sample = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                )
                assert tokens[row["id"]] == {
                    "ids": sample.input_ids[0].tolist(),
                    "prompt_length": sample.prompt_length,
                }
    fits = json.loads((results / "fits.json").read_text())
    requests = {(r["case"], r["condition"]["id"]): r for r in protocol["requests"]}
    assert {(r["case"], r["condition"]["id"]) for r in fits} == set(requests)
    assert len(fits) == len(requests) == protocol["new_fits"]
    observations = 0
    teacher_records = 0
    adam_records = 0
    controls = 2 * len({r["case"] for r in fits})
    for fit in fits:
        c, name = fit["case"], fit["condition"]["id"]
        prefix = f"case-{c}-{name}"
        request = requests[(c, name)]
        case = cases[c]
        old = oldfits[(c, name)]
        assert fit["condition"] == request["condition"] and fit["role"] == request["role"]
        assert fit["history"] == old["history"] and len(fit["history"]) == protocol["steps"]
        assert (
            fit["final_history_exact"]
            and fit["final_parameters_exact"]
            and fit["final_live_teacher_exact"]
            and fit["base_restored"]
        )
        assert fit["trainable_scalars"] == request["trainable_scalars"]
        tensor_name = (
            "adapter_model.safetensors"
            if fit["condition"]["family"] == "lora"
            else "patch.safetensors"
        )
        a = load_file(str(results / f"{prefix}-full-adapter" / tensor_name))
        b = load_file(str(original / f"{prefix}-adapter" / tensor_name))
        assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
        assert sum(v.numel() for v in a.values()) == fit["trainable_scalars"]
        final_live = load_file(str(results / f"{prefix}-final-live-teacher.safetensors"))
        parent_live = load_file(str(replay / f"{prefix}-live-teacher.safetensors"))
        assert all(torch.equal(final_live[k], parent_live[k]) for k in final_live)
        rows = [r for bank in ("train", "preserve") for r in case[bank]]
        labels = []
        lengths = []
        for row in rows:
            item = tokens[row["id"]]
            answer = item["ids"][item["prompt_length"] :]
            labels.extend(answer)
            lengths.append(len(answer))
        snapshots = json.loads((results / f"{prefix}-checkpoints.json").read_text())
        assert snapshots == fit["checkpoints"]
        assert [r["step"] for r in snapshots] == protocol["checkpoint_steps"]
        first = None
        last_time = last_monitor = 0
        for snap in snapshots:
            step = snap["step"]
            t = load_file(str(results / snap["teacher_file"]))
            assert t["labels"].tolist() == labels and t["lengths"].tolist() == lengths
            expected = _calibration(t, len(case["train"]), protocol["gate_probability"])
            assert expected["adequate"] == snap["calibration"]["adequate"]
            for k, v in expected.items():
                if k != "adequate":
                    _close(v, snap["calibration"][k])
            if expected["adequate"] and first is None:
                first = step
            assert (
                snap["training_seconds"] >= last_time and snap["monitoring_seconds"] >= last_monitor
            )
            last_time = snap["training_seconds"]
            last_monitor = snap["monitoring_seconds"]
            if "mode_comparison" in snap:
                g = load_file(str(results / f"{prefix}-step-{step}-grad-calibration.safetensors"))
                _compare(g, t, snap["mode_comparison"])
                teacher_records += 1
            teacher_records += 1
        assert first == fit["selected_step"]
        assert set(fit["states"]) == ({"full", "early"} if first is not None else {"full"})
        for state, record in fit["states"].items():
            directory = results / f"{prefix}-{state}-adapter"
            weights = load_file(str(directory / tensor_name))
            assert set(weights) == set(a)
            assert all(
                v.dtype == torch.float32 and torch.isfinite(v).all() for v in weights.values()
            )
            assert all(weights[k].shape == a[k].shape for k in weights)
            assert sum(v.numel() for v in weights.values()) == fit["trainable_scalars"]
            if fit["condition"]["family"] == "lora":
                config = json.loads((directory / "adapter_config.json").read_text())
                assert config["r"] == config["lora_alpha"] == protocol["rank"] == 1
                assert config["bias"] == "none" and config["lora_dropout"] == 0
                squared = sum(
                    float(v.double().square().sum())
                    * float(
                        weights[k.replace("lora_A.weight", "lora_B.weight")].double().square().sum()
                    )
                    for k, v in weights.items()
                    if k.endswith("lora_A.weight")
                )
            else:
                config = json.loads((directory / "patch.json").read_text())
                assert config["base_fingerprint"] == protocol["base_fingerprint"]
                assert config["selections"] == request["supports"] and config["train_incoming"]
                squared = sum(float(v.double().square().sum()) for v in weights.values())
            _close(record["nominal_prequantization_update_l2"], math.sqrt(squared))
            step = first if state == "early" else protocol["steps"]
            assert record["step"] == step and record["compact_reload_teacher_exact"]
            t = load_file(str(results / f"{prefix}-{state}-live-teacher.safetensors"))
            checkpoint = load_file(str(results / f"{prefix}-step-{step}-calibration.safetensors"))
            assert torch.equal(t["logits"][: len(labels)], checkpoint["logits"])
            assert (
                t["labels"][: len(labels)].tolist() == labels
                and t["lengths"][: len(lengths)].tolist() == lengths
            )
            merged = load_file(str(results / f"{prefix}-{state}-merged-teacher.safetensors"))
            assert torch.equal(merged["labels"], t["labels"]) and torch.equal(
                merged["lengths"], t["lengths"]
            )
            parts = _losses(merged)
            n = len(case["train"])
            p = len(case["preserve"])
            expected = {
                "task_mean_nll": sum(float(x.mean()) for x in parts[:n]) / n,
                "task_worst_token_nll": max(float(x.max()) for x in parts[:n]),
                "preserve_mean_nll": sum(float(x.mean()) for x in parts[n : n + p]) / p,
                "target_val_mean_nll": sum(float(x.mean()) for x in parts[n + p :])
                / len(parts[n + p :]),
            }
            for k, v in expected.items():
                _close(v, record["metrics"][k])
            short = {
                "logits": merged["logits"][: len(labels)],
                "labels": merged["labels"][: len(labels)],
                "lengths": merged["lengths"][: len(lengths)],
            }
            expected = _calibration(short, n, protocol["gate_probability"])
            assert record["merged_calibration"]["adequate"] == expected["adequate"]
            for k, v in expected.items():
                if k != "adequate":
                    _close(v, record["merged_calibration"][k])
            generated = json.loads((results / f"{prefix}-{state}.json").read_text())
            assert set(generated) == set(protocol["generation_banks"])
            for bank in protocol["generation_banks"]:
                for row, answer in zip(case[bank], generated[bank], strict=True):
                    assert row["id"] == answer["id"]
                    ids = answer["tokens"]
                    assert 0 < len(ids) <= protocol["max_new_tokens"]
                    assert tokenizer.eos_token_id not in ids[:-1]
                    assert answer["text"] == tokenizer.decode(ids, skip_special_tokens=True)
                    ended = ids[-1] == tokenizer.eos_token_id
                    assert answer["ended"] == ended and answer["truncated"] == (not ended)
                    assert ended or len(ids) == protocol["max_new_tokens"]
                    assert answer["unexpected_special"] == any(
                        i in tokenizer.all_special_ids and i != tokenizer.eos_token_id for i in ids
                    )
                    observations += 1
            if state == "full":
                parent = load_file(str(original / f"{prefix}-teacher.safetensors"))
                assert all(torch.equal(merged[k], parent[k]) for k in merged)
                old_outputs = json.loads((original / f"{prefix}.json").read_text())
                assert all(generated[b] == old_outputs[b] for b in protocol["generation_banks"])
                assert (
                    record["parent_merged_teacher_exact"]
                    and record["parent_merged_generations_exact"]
                )
            controls += 2
            teacher_records += 2
        if request["role"] == "known_failure_diagnostic":
            audit = json.loads((results / f"{prefix}-final-update.json").read_text())
            values = load_file(str(results / f"{prefix}-final-update.safetensors"))
            tensors = {
                s: load_file(str(results / f"{prefix}-before-final-{s}.safetensors"))
                for s in ("grad", "no-grad")
            }
            _compare(tensors["grad"], tensors["no-grad"], audit["pre_update_mode_comparison"])
            for state, key in (
                ("grad", "pre_update_grad_task_nll"),
                ("no-grad", "pre_update_no_grad_task_nll"),
            ):
                _close(float(torch.cat(_losses(tensors[state])).mean()), audit[key])
            assert (
                audit["step"] == protocol["steps"]
                and audit["example_indices"] == fit["history"][-1]["example_indices"]
            )
            _close(audit["pre_update_logged_task_nll"], fit["history"][-1]["task_nll"])
            index = audit["example_indices"][0]
            for mode, key in (
                ("grad", "post_update_grad_task_nll"),
                ("no-grad", "post_update_no_grad_task_nll"),
            ):
                suffix = "grad-calibration" if mode == "grad" else "calibration"
                path = results / f"{prefix}-step-{protocol['steps']}-{suffix}.safetensors"
                t = load_file(str(path))
                _close(float(_losses(t)[index].mean()), audit[key])
            joint_dot = task_dot = update_sq = joint_sq = task_sq = 0.0
            group = audit["optimizer_group"]
            beta1, beta2 = group["betas"]
            step = audit["step"]
            assert group["weight_decay"] == 0
            for i, name in enumerate(audit["parameter_names"]):
                before = values[f"parameter-before/{i}"].double()
                after = values[f"parameter-after/{i}"].double()
                g = values[f"clipped-joint-gradient/{i}"].double()
                task = values[f"correction-gradient/{i}"].double()
                assert torch.equal(values[f"parameter-after/{i}"], a[name.removesuffix(".delta")])
                update = after - before
                joint_dot += float((g * update).sum())
                task_dot += float((task * update).sum())
                update_sq += float(update.square().sum())
                joint_sq += float(g.square().sum())
                task_sq += float(task.square().sum())
                m = beta1 * values[f"adam-first-moment/{i}"].double() + (1 - beta1) * g
                v = beta2 * values[f"adam-second-moment/{i}"].double() + (1 - beta2) * g.square()
                predicted = before - group["lr"] * (m / (1 - beta1**step)) / (
                    (v / (1 - beta2**step)).sqrt() + group["eps"]
                )
                torch.testing.assert_close(after, predicted, rtol=2e-6, atol=1e-8)
            for key, value in (
                ("clipped_joint_gradient_dot_update", joint_dot),
                ("correction_gradient_dot_update", task_dot),
                ("update_l2", math.sqrt(update_sq)),
                ("clipped_joint_gradient_l2", math.sqrt(joint_sq)),
                ("correction_gradient_l2", math.sqrt(task_sq)),
            ):
                _close(audit[key], value, 1e-12)
            adam_records += 1
            teacher_records += 2
    completion = json.loads((results / "completion.json").read_text())
    assert completion == {
        "status": "complete",
        "new_fits": len(fits),
        "observations": observations,
        "base_generation_controls": controls,
        "base_restored": True,
        "final_fingerprint": protocol["base_fingerprint"],
    }
    return {
        "complete": True,
        "fits": len(fits),
        "decoded": observations,
        "teacher_records_recomputed": teacher_records,
        "adam_steps_reconstructed": adam_records,
        "all_parent_final_controls_exact": True,
        "file_sha256": {
            str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in results.rglob("*")
            if p.is_file()
        },
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("original", "replay", "results", "protocol", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    proof = verify(
        {"original": args.original, "replay": args.replay}, args.results, protocol, tokenizer
    )
    with args.output.open("x") as out:
        out.write(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "file_sha256"}))


if __name__ == "__main__":
    main()
