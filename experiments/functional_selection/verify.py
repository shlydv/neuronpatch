"""Recompute selection,update arithmetic and conditional-logit measurements offline."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from experiments.paired_endpoint.run import verify_parent
from experiments.paired_mlp.data import messages
from neuronpatch.supervision import chat_response


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-9, rel_tol=1e-7), (a, b)


def checked_metrics(base, edited, task_count, calibration_count):
    assert torch.equal(base["labels"], edited["labels"])
    assert torch.equal(base["lengths"], edited["lengths"])
    labels, lengths = base["labels"], base["lengths"].tolist()
    old = F.cross_entropy(base["logits"].double(), labels, reduction="none").split(lengths)
    new = F.cross_entropy(edited["logits"].double(), labels, reduction="none").split(lengths)
    divergence = (
        F.kl_div(
            edited["logits"].double().log_softmax(-1),
            base["logits"].double().log_softmax(-1),
            log_target=True,
            reduction="none",
        )
        .sum(-1)
        .split(lengths)
    )
    boundary = task_count + calibration_count
    return {
        "task_base_mean_nll": sum(float(p.mean()) for p in old[:task_count]) / task_count,
        "task_mean_nll": sum(float(p.mean()) for p in new[:task_count]) / task_count,
        "task_nll_improvement": sum(
            float(a.mean() - b.mean())
            for a, b in zip(old[:task_count], new[:task_count], strict=True)
        )
        / task_count,
        "calibration_mean_anchored_kl": sum(
            float(p.mean()) for p in divergence[task_count:boundary]
        )
        / calibration_count,
        "calibration_worst_token_anchored_kl": max(
            float(p.max()) for p in divergence[task_count:boundary]
        ),
        "heldout_mean_anchored_kl": sum(float(p.mean()) for p in divergence[boundary:])
        / len(divergence[boundary:]),
        "heldout_worst_token_anchored_kl": max(float(p.max()) for p in divergence[boundary:]),
        "changed_teacher_argmax_positions": int(
            (base["logits"].argmax(-1) != edited["logits"].argmax(-1)).sum()
        ),
    }


def verify(parent, results, protocol, tokenizer):
    verify_parent(parent, protocol)
    assert json.loads((results / "protocol.json").read_text()) == protocol
    identity = json.loads((results / "identity.json").read_text())
    assert identity["base_fingerprint"] == protocol["base_fingerprint"]
    counts = identity["layer_counts"]
    cases = json.loads((parent / "inputs.json").read_text())["cases"]
    records = json.loads((results / "records.json").read_text())
    expected = {
        (c["case"], condition, scale)
        for c in cases
        for condition in protocol["conditions"]
        for scale in protocol["scales"]
    }
    assert {(r["case"], r["condition"], r["scale"]) for r in records} == expected
    assert len(records) == len(expected) == protocol["expected_one_step_edits"]
    observations = sum(len(c[b]) for c in cases for b in protocol["generation_banks"])
    controls = 0
    teacher_records = 0
    for case in cases:
        c = case["case"]
        baseline = load_file(str(results / f"case-{c}-base-teacher.safetensors"))
        rows = [
            r for b in ("train", "preserve", *protocol["heldout_teacher_banks"]) for r in case[b]
        ]
        labels, lengths = [], []
        for row in rows:
            sample = chat_response(
                tokenizer,
                messages(row),
                row["answers"][0],
                max_length=protocol["max_sequence_tokens"],
            )
            response = sample.input_ids[0, sample.prompt_length :].tolist()
            labels.extend(response)
            lengths.append(len(response))
        assert baseline["labels"].tolist() == labels and baseline["lengths"].tolist() == lengths
        saved = load_file(str(results / f"case-{c}-selection.safetensors"))
        discovery = json.loads((results / f"case-{c}-discovery.json").read_text())
        candidates = discovery["candidates"]
        old_scores = load_file(str(parent / f"case-{c}-scores.safetensors"))
        generator = torch.Generator().manual_seed(protocol["random_support_seed"])
        eligible, random_mask, magnitude = [], [], []
        for i, (group, count) in enumerate(zip(candidates, counts, strict=True)):
            original = old_scores[f"{i}-paired_gradient_norm"]
            top = set(
                torch.argsort(original, descending=True, stable=True)[
                    : protocol["shortlist_per_layer"]
                ].tolist()
            )
            draw = set(torch.randperm(len(original), generator=generator)[:count].tolist())
            assert group["indices"] == sorted(top | draw)
            eligible.extend(v in top for v in group["indices"])
            random_mask.extend(v in draw for v in group["indices"])
            magnitude.append(original[torch.tensor(group["indices"])])
        assert saved["eligible"].tolist() == eligible
        assert saved["selected/random"].tolist() == random_mask
        assert torch.equal(saved["gradient_norm"], torch.cat(magnitude))
        allowed = saved["eligible"]
        response = saved["responses"].double()
        assert len(response) == len(case["preserve"]) * protocol["fisher_probes_per_anchor"]
        risk = response[:, allowed].square().mean(0)
        norms = saved["direction_squared_norm"][allowed]
        damping = (
            float(protocol["damping_ratio"] * risk.sum() / norms.sum()) if norms.sum() > 0 else 0
        )
        close(damping, discovery["damping"])
        denominator = risk + damping * norms if risk.sum() > 0 else norms
        ranked = torch.zeros_like(saved["gain"][allowed])
        valid = denominator > 0
        ranked[valid] = saved["gain"][allowed][valid].square() / denominator[valid]
        torch.testing.assert_close(ranked, saved["risk_scores"][allowed], rtol=1e-12, atol=1e-12)
        for condition, values in [
            ("gradient_norm", saved["gradient_norm"]),
            ("adam_gain", saved["gain"]),
            ("gain_risk", saved["risk_scores"]),
        ]:
            expected_mask = torch.zeros_like(allowed)
            offset = 0
            for group, count in zip(candidates, counts, strict=True):
                local = [j for j in range(len(group["indices"])) if eligible[offset + j]]
                best = sorted(
                    local, key=lambda j: (-float(values[offset + j]), group["indices"][j])
                )[:count]
                expected_mask[offset + torch.tensor(best)] = True
                offset += len(group["indices"])
            assert torch.equal(expected_mask, saved[f"selected/{condition}"])
        for record in [r for r in records if r["case"] == c]:
            condition, scale, prefix = record["condition"], record["scale"], record["prefix"]
            mask = saved[f"selected/{condition}"]
            selected_layout = []
            offset = 0
            for group in candidates:
                width = len(group["indices"])
                selected_layout.append(
                    {
                        "incoming": group["incoming"],
                        "outgoing": group["outgoing"],
                        "indices": [
                            v
                            for v, picked in zip(
                                group["indices"],
                                mask[offset : offset + width].tolist(),
                                strict=True,
                            )
                            if picked
                        ],
                    }
                )
                offset += width
            assert record["selections"] == selected_layout and record["base_slices_restored_exact"]
            tensors = load_file(str(results / f"case-{c}-{condition}-direction.safetensors"))
            applied = load_file(str(results / f"{prefix}-applied.safetensors"))
            gain = nominal_squared = actual_squared = 0.0
            scalar_count = 0
            for group, count in zip(selected_layout, counts, strict=True):
                assert len(group["indices"]) == count
                proposals = {}
                for name in (*group["incoming"], group["outgoing"]):
                    gradient = tensors[f"gradient/{name}"].double()
                    proposals[name] = (
                        -gradient
                        if protocol["direction"] == "gradient"
                        else -gradient / (gradient.abs() + protocol["direction_epsilon"])
                    )
                norm = (
                    sum(
                        t.square().sum(1 if name in group["incoming"] else 0)
                        for name, t in proposals.items()
                    )
                    .sqrt()
                    .clamp_min(torch.finfo(torch.float64).tiny)
                )
                for name, proposal in proposals.items():
                    expected_direction = proposal / (
                        norm[:, None] if name in group["incoming"] else norm[None, :]
                    )
                    direction = tensors[f"direction/{name}"]
                    torch.testing.assert_close(
                        direction, expected_direction, rtol=1e-12, atol=1e-12
                    )
                    base = tensors[f"base/{name}"]
                    dtype = torch.promote_types(base.dtype, torch.float32)
                    expected_weight = (base.to(dtype) + scale * direction.to(dtype)).to(base.dtype)
                    assert torch.equal(expected_weight, applied[name])
                    gain -= float((tensors[f"gradient/{name}"].double() * direction).sum())
                    nominal_squared += float(direction.square().sum()) * scale**2
                    actual_squared += float((applied[name].double() - base.double()).square().sum())
                    scalar_count += direction.numel()
            assert (
                scalar_count
                == record["changed_scalar_coordinates"]
                == identity["budgets"]["paired"]
            )
            close(math.sqrt(nominal_squared), record["nominal_update_l2"])
            close(math.sqrt(actual_squared), record["actual_quantized_update_l2"])
            close(scale * gain, record["predicted_correction_nll_improvement"])
            close(
                0.5 * scale**2 * float(response[:, mask].sum(1).square().mean()),
                record["predicted_calibration_mean_anchored_kl"],
            )
            close(
                0.5 * scale**2 * float(response[:, mask].square().mean(0).sum()),
                record["diagonal_calibration_mean_anchored_kl"],
            )
            edited = load_file(str(results / f"{prefix}-teacher.safetensors"))
            expected_metrics = checked_metrics(
                baseline, edited, len(case["train"]), len(case["preserve"])
            )
            for key, value in expected_metrics.items():
                close(value, record["metrics"][key])
            teacher_records += 1
            assert record["generated"] == (scale == protocol["generation_scale"])
            if record["generated"]:
                outputs = json.loads((results / f"{prefix}-generations.json").read_text())
                assert set(outputs) == set(protocol["generation_banks"])
                for b in protocol["generation_banks"]:
                    for row, answer in zip(case[b], outputs[b], strict=True):
                        assert row["id"] == answer["id"]
                        ids = answer["tokens"]
                        assert (
                            0 < len(ids) <= protocol["max_new_tokens"]
                            and tokenizer.eos_token_id not in ids[:-1]
                        )
                        assert answer["text"] == tokenizer.decode(ids, skip_special_tokens=True)
                        ended = ids[-1] == tokenizer.eos_token_id
                        assert ended == answer["ended"] and answer["truncated"] == (not ended)
                        assert ended or len(ids) == protocol["max_new_tokens"]
                        assert answer["unexpected_special"] == any(
                            t in tokenizer.all_special_ids and t != tokenizer.eos_token_id
                            for t in ids
                        )
                        observations += 1
            controls += min(2, len(case[protocol["generation_banks"][0]]))
        teacher_records += 1
    completion = json.loads((results / "completion.json").read_text())
    assert completion == {
        "status": "complete",
        "optimizer_trajectories": 0,
        "one_step_edits": len(records),
        "observations": observations,
        "base_generation_controls": controls,
        "final_fingerprint": protocol["base_fingerprint"],
        "base_restored": True,
    }
    return {
        "complete": True,
        "one_step_edits": len(records),
        "decoded": observations,
        "teacher_records_recomputed": teacher_records,
        "scope": (
            "Saved direction/weight arithmetic,selection rankings,teacher metrics and decoding "
            "verified. Probe autograd correctness is tested separately against dense oracles; "
            "no fresh method advantage follows."
        ),
        "file_sha256": {
            str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in results.rglob("*")
            if p.is_file()
        },
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    result = verify(args.parent, args.results, protocol, tokenizer)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
