"""Calibrate functional-risk predictions on finite,exactly restored Qwen updates."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.paired_endpoint.run import find_parent, verify_parent
from experiments.paired_mlp.data import messages
from experiments.paired_mlp.model import layout, supports
from experiments.paired_mlp.run import generate, write
from experiments.paired_mlp.selection import capture_response_factors, projection_scores
from experiments.paired_mlp.training import teacher
from neuronpatch.paired import MLPSelection
from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .directions import DirectionBank, build_directions, probe_dots, rank_scores
from .fisher import capture_fisher_factors, restrict_factors
from .perturb import perturb


def shortlist(groups, counts, scores, limit, random_seed):
    random = supports(groups, counts, scores, family="paired_random", seed=random_seed)
    selections, eligible, gradient, random_mask = [], [], [], []
    for group, draw, score in zip(groups, random, scores, strict=True):
        magnitude = score["paired_gradient_norm"]
        top = set(torch.argsort(magnitude, descending=True, stable=True)[:limit].tolist())
        indices = tuple(sorted(top | set(draw.indices)))
        selections.append(MLPSelection(group["incoming"], group["outgoing"], indices))
        eligible.extend(i in top for i in indices)
        random_mask.extend(i in draw.indices for i in indices)
        gradient.append(magnitude[torch.tensor(indices)])
    return selections, torch.tensor(eligible), torch.cat(gradient), torch.tensor(random_mask)


def choose(bank, values, eligible, counts):
    selected = torch.zeros_like(eligible)
    offset = 0
    for selection, count in zip(bank.selections, counts, strict=True):
        end = offset + len(selection.indices)
        allowed = eligible[offset:end].nonzero().flatten()
        assert len(allowed) >= count
        order = torch.argsort(values[offset:end][allowed], descending=True, stable=True)
        selected[offset + allowed[order[:count]]] = True
        offset = end
    assert offset == len(selected) and int(selected.sum()) == sum(counts)
    return selected


def selected_layout(bank, mask):
    result = []
    offset = 0
    for selection in bank.selections:
        end = offset + len(selection.indices)
        indices = [
            index
            for index, picked in zip(selection.indices, mask[offset:end].tolist(), strict=True)
            if picked
        ]
        result.append(
            {"incoming": selection.incoming, "outgoing": selection.outgoing, "indices": indices}
        )
        offset = end
    return result


def metrics(base, edited, task_count, calibration_count):
    assert torch.equal(base["labels"], edited["labels"])
    assert torch.equal(base["lengths"], edited["lengths"])
    before = base["logits"].double().log_softmax(-1)
    after = edited["logits"].double().log_softmax(-1)
    labels, lengths = base["labels"], base["lengths"].tolist()
    old_nll = (-before.gather(1, labels[:, None])[:, 0]).split(lengths)
    new_nll = (-after.gather(1, labels[:, None])[:, 0]).split(lengths)
    kl = (before.exp() * (before - after)).sum(-1).split(lengths)
    assert all(torch.isfinite(part).all() and float(part.min()) >= -1e-10 for part in kl)
    boundary = task_count + calibration_count
    return {
        "task_base_mean_nll": sum(float(v.mean()) for v in old_nll[:task_count]) / task_count,
        "task_mean_nll": sum(float(v.mean()) for v in new_nll[:task_count]) / task_count,
        "task_nll_improvement": sum(
            float(a.mean() - b.mean())
            for a, b in zip(old_nll[:task_count], new_nll[:task_count], strict=True)
        )
        / task_count,
        "calibration_mean_anchored_kl": sum(float(v.mean()) for v in kl[task_count:boundary])
        / calibration_count,
        "calibration_worst_token_anchored_kl": max(float(v.max()) for v in kl[task_count:boundary]),
        "heldout_mean_anchored_kl": sum(float(v.mean()) for v in kl[boundary:])
        / len(kl[boundary:]),
        "heldout_worst_token_anchored_kl": max(float(v.max()) for v in kl[boundary:]),
        "changed_teacher_argmax_positions": int((before.argmax(-1) != after.argmax(-1)).sum()),
    }


def execute(model, tokenizer, parent, protocol, output, check_time):
    verify_parent(parent, protocol)
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    identity = fingerprint(model)
    assert identity == protocol["base_fingerprint"]
    data = json.loads((parent / "inputs.json").read_text())
    tokenized = json.loads((parent / "tokenization.json").read_text())
    groups, counts, budgets = layout(model, protocol["rank_for_budget"])
    assert {
        "layers": len(groups),
        "hidden": groups[0]["hidden"],
        "intermediate": groups[0]["intermediate"],
    } == protocol["architecture"]
    names = [name for group in groups for name in (*group["incoming"], group["outgoing"])]
    write(output / "protocol.json", protocol)
    write(
        output / "identity.json",
        {
            "base_fingerprint": identity,
            "budgets": budgets,
            "layer_counts": counts,
            "optimizer_trajectories": 0,
        },
    )
    examples, bases, records = {}, {}, []
    observations = controls = 0
    device = next(model.parameters()).device
    for case in data["cases"]:
        c = case["case"]
        print("CASE_START", c, flush=True)
        for bank in ("train", "preserve", *protocol["generation_banks"]):
            for row in case[bank]:
                sample = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                    device=device,
                )
                assert tokenized[row["id"]] == {
                    "ids": sample.input_ids[0].tolist(),
                    "prompt_length": sample.prompt_length,
                }
                examples[row["id"]] = sample
        task = [examples[row["id"]] for row in case["train"]]
        calibration = [examples[row["id"]] for row in case["preserve"]]
        measured = [
            *task,
            *calibration,
            *[
                examples[row["id"]]
                for bank in protocol["heldout_teacher_banks"]
                for row in case[bank]
            ],
        ]
        base_teacher = teacher(model, measured)
        save_file(base_teacher, output / f"case-{c}-base-teacher.safetensors")
        parent_base = json.loads((parent / f"case-{c}-base.json").read_text())
        for bank in protocol["generation_banks"]:
            for row, expected in zip(case[bank], parent_base[bank], strict=True):
                check_time()
                if row["id"] not in bases:
                    bases[row["id"]] = generate(
                        model, tokenizer, examples[row["id"]], row, protocol["max_new_tokens"]
                    )
                assert bases[row["id"]] == expected
                observations += 1
        started = time.monotonic()
        captures = []
        for sample in task:
            check_time()
            captures.append(
                capture_response_factors(model, sample, names, loss_scale=protocol["loss_scale"])[0]
            )
        scores = [
            projection_scores(captures, incoming=g["incoming"], outgoing=g["outgoing"])
            for g in groups
        ]
        old_scores = load_file(str(parent / f"case-{c}-scores.safetensors"))
        for i, score in enumerate(scores):
            torch.testing.assert_close(
                score["paired_gradient_norm"],
                old_scores[f"{i}-paired_gradient_norm"],
                rtol=1e-6,
                atol=1e-10,
            )
            # Preserve the exact original candidate ordering even at a near tie.
            score["paired_gradient_norm"] = old_scores[f"{i}-paired_gradient_norm"]
        candidates, eligible, magnitude, random_mask = shortlist(
            groups, counts, scores, protocol["shortlist_per_layer"], protocol["random_support_seed"]
        )
        bank = build_directions(
            restrict_factors(captures, candidates),
            candidates,
            kind=protocol["direction"],
            epsilon=protocol["direction_epsilon"],
        )
        del captures
        proposal_seconds = time.monotonic() - started
        started = time.monotonic()
        responses = []
        for i, sample in enumerate(calibration):
            check_time()
            probes = capture_fisher_factors(
                model,
                sample,
                candidates,
                probes=protocol["fisher_probes_per_anchor"],
                seed=protocol["seed"] + 1000 * c + i,
                loss_scale=protocol["loss_scale"],
            )
            responses.extend(probe_dots(bank, probe) for probe in probes)
            del probes
        responses = torch.stack(responses)
        eligible_bank = DirectionBank(
            (), {}, bank.gain[eligible], bank.squared_norm[eligible], bank.kind
        )
        ranked = rank_scores(
            eligible_bank, responses[:, eligible], damping_ratio=protocol["damping_ratio"]
        )
        risk_values = torch.zeros_like(bank.gain)
        risk_values[eligible] = ranked["scores"]
        risk_seconds = time.monotonic() - started
        masks = {
            "gradient_norm": choose(bank, magnitude, eligible, counts),
            "adam_gain": choose(bank, bank.gain, eligible, counts),
            "gain_risk": choose(bank, risk_values, eligible, counts),
            "random": random_mask,
        }
        save_file(
            {
                "eligible": eligible,
                "gradient_norm": magnitude,
                "gain": bank.gain,
                "direction_squared_norm": bank.squared_norm,
                "responses": responses,
                "risk_scores": risk_values,
                **{f"selected/{key}": value for key, value in masks.items()},
            },
            output / f"case-{c}-selection.safetensors",
        )
        discovery = {
            "case": c,
            "proposal_seconds": proposal_seconds,
            "risk_seconds": risk_seconds,
            "correction_backwards": len(task),
            "fisher_forwards": len(calibration),
            "fisher_backwards": len(responses),
            "damping": ranked["damping"],
            "risk_informative": ranked["risk_informative"],
            "candidates": [
                {"incoming": s.incoming, "outgoing": s.outgoing, "indices": s.indices}
                for s in candidates
            ],
        }
        write(output / f"case-{c}-discovery.json", discovery)
        print(
            "DISCOVERY_COMPLETE",
            c,
            {k: v for k, v in discovery.items() if k != "candidates"},
            flush=True,
        )
        for condition in protocol["conditions"]:
            mask = masks[condition]
            assert int(mask.sum()) == sum(counts) and float(bank.gain[mask].sum()) > 0
            support = selected_layout(bank, mask)
            saved_direction = False
            for scale in protocol["scales"]:
                check_time()
                prefix = f"case-{c}-{condition}-{scale:.3f}"
                with perturb(model, bank, mask, scale) as changed:
                    if not saved_direction:
                        save_file(
                            {
                                **{f"base/{n}": t for n, t in changed["base"].items()},
                                **{f"direction/{n}": t for n, t in changed["directions"].items()},
                                **{f"gradient/{n}": t for n, t in changed["gradients"].items()},
                            },
                            output / f"case-{c}-{condition}-direction.safetensors",
                        )
                        saved_direction = True
                    save_file(changed["after"], output / f"{prefix}-applied.safetensors")
                    edited = teacher(model, measured)
                    save_file(edited, output / f"{prefix}-teacher.safetensors")
                    measured_metrics = metrics(base_teacher, edited, len(task), len(calibration))
                    generated = None
                    if scale == protocol["generation_scale"]:
                        generated = {}
                        for target_bank in protocol["generation_banks"]:
                            generated[target_bank] = []
                            for row in case[target_bank]:
                                check_time()
                                generated[target_bank].append(
                                    generate(
                                        model,
                                        tokenizer,
                                        examples[row["id"]],
                                        row,
                                        protocol["max_new_tokens"],
                                    )
                                )
                                observations += 1
                        write(output / f"{prefix}-generations.json", generated)
                for row in case[protocol["generation_banks"][0]][:2]:
                    check_time()
                    actual = generate(
                        model, tokenizer, examples[row["id"]], row, protocol["max_new_tokens"]
                    )
                    assert actual == bases[row["id"]]
                    controls += 1
                record = {
                    "case": c,
                    "condition": condition,
                    "scale": scale,
                    "prefix": prefix,
                    "selections": support,
                    "changed_scalar_coordinates": sum(t.numel() for t in changed["after"].values()),
                    "nominal_update_l2": changed["nominal_update_l2"],
                    "actual_quantized_update_l2": changed["actual_quantized_update_l2"],
                    "predicted_correction_nll_improvement": scale * float(bank.gain[mask].sum()),
                    "predicted_calibration_mean_anchored_kl": 0.5
                    * scale**2
                    * float(responses[:, mask].sum(1).square().mean()),
                    "diagonal_calibration_mean_anchored_kl": 0.5
                    * scale**2
                    * float(responses[:, mask].square().mean(0).sum()),
                    "metrics": measured_metrics,
                    "base_slices_restored_exact": True,
                    "generated": generated is not None,
                }
                records.append(record)
                write(output / "records.json", records)
                print("FINITE_STEP", c, condition, scale, measured_metrics, flush=True)
                del edited, changed, generated
        del bank, responses, base_teacher
        gc.collect()
    assert fingerprint(model) == identity
    assert len(records) == protocol["expected_one_step_edits"]
    assert observations == protocol["expected_scored_observations"]
    assert controls == protocol["expected_base_generation_controls"]
    write(
        output / "completion.json",
        {
            "status": "complete",
            "optimizer_trajectories": 0,
            "one_step_edits": len(records),
            "observations": observations,
            "base_generation_controls": controls,
            "final_fingerprint": identity,
            "base_restored": True,
        },
    )
    print("COMPLETE functional-direction-calibration", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    versions = {name: importlib.metadata.version(name) for name in protocol["runtime"]}
    assert all(versions[name].split("+")[0] == value for name, value in protocol["runtime"].items())
    parent = find_parent(Path("/kaggle/input"), protocol)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded functional-direction calibration exceeded runtime")

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
