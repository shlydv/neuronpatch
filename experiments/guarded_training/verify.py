"""Independently check final logits, decoding, artifacts and recorded acceptance rules."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.functional_selection.verify import checked_metrics, close
from experiments.paired_endpoint.run import verify_parent
from experiments.paired_mlp.data import messages
from neuronpatch.supervision import chat_response

from .step import Budget, Measurement


def verify(parent, results, protocol, tokenizer):
    verify_parent(parent, protocol)
    assert json.loads((results / "protocol.json").read_text()) == protocol
    cases = json.loads((parent / "inputs.json").read_text())["cases"]
    fits = json.loads((results / "fits.json").read_text())
    assert len(fits) == protocol["expected_fits"]
    assert {(r["case"], r["method"]) for r in fits} == {
        (c["case"], m) for c in cases for m in protocol["methods"]
    }
    decoded = teachers = controls = 0
    budget = Budget(**protocol["budget"])

    def decode(case, outputs):
        nonlocal decoded
        assert set(outputs) == set(protocol["generation_banks"])
        for bank in protocol["generation_banks"]:
            for row, answer in zip(case[bank], outputs[bank], strict=True):
                assert row["id"] == answer["id"]
                ids = answer["tokens"]
                assert (
                    0 < len(ids) <= protocol["max_new_tokens"]
                    and tokenizer.eos_token_id not in ids[:-1]
                )
                assert answer["text"] == tokenizer.decode(ids, skip_special_tokens=True)
                ended = ids[-1] == tokenizer.eos_token_id
                assert answer["ended"] == ended and answer["truncated"] == (not ended)
                assert ended or len(ids) == protocol["max_new_tokens"]
                assert answer["unexpected_special"] == any(
                    t in tokenizer.all_special_ids and t != tokenizer.eos_token_id for t in ids
                )
                decoded += 1

    for case in cases:
        c = case["case"]
        baseline = json.loads((results / f"case-{c}-base.json").read_text())
        original = json.loads((parent / f"case-{c}-base.json").read_text())
        assert all(baseline[b] == original[b] for b in protocol["generation_banks"])
        decode(case, baseline)
        base = load_file(str(results / f"case-{c}-base-teacher.safetensors"))
        rows = [
            r
            for bank in ("train", "preserve", *protocol["heldout_teacher_banks"])
            for r in case[bank]
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
        assert base["labels"].tolist() == labels and base["lengths"].tolist() == lengths
        teachers += 1
        for fit in [r for r in fits if r["case"] == c]:
            prefix = f"case-{c}-{fit['method']}"
            assert (
                fit["base_restored"]
                and fit["reload_teacher_exact"]
                and fit["reload_generations_exact"]
            )
            history = fit["fit"]["history"]
            assert len(history) <= protocol["steps"]
            before = Measurement(**fit["fit"]["initial"])
            accepted = 0
            for index, row in enumerate(history):
                assert row["step"] == index + 1 and math.isfinite(row["gradient_norm"])
                attempts = row["attempts"]
                assert 0 < len(attempts) <= len(protocol["backtrack_scales"])
                assert [a["scale"] for a in attempts] == protocol["backtrack_scales"][
                    : len(attempts)
                ]
                for j, attempt in enumerate(attempts):
                    current = Measurement(**attempt["measurement"])
                    admissible = (
                        current.task_nll <= before.task_nll - budget.minimum_improvement
                        and current.mean_kl <= budget.mean_kl
                        and current.worst_kl <= budget.worst_kl
                    )
                    assert attempt["accepted"] == admissible
                    assert not attempt["accepted"] or j == len(attempts) - 1
                assert row["accepted"] == attempts[-1]["accepted"]
                if row["accepted"]:
                    before = Measurement(**attempts[-1]["measurement"])
                    accepted += 1
                else:
                    assert index == len(history) - 1 and len(attempts) == len(
                        protocol["backtrack_scales"]
                    )
            assert accepted == fit["fit"]["accepted_updates"]
            assert before.__dict__ == fit["fit"]["final"]
            directory = results / f"{prefix}-adapter"
            tensors = load_file(
                str(
                    directory
                    / (
                        "adapter_model.safetensors"
                        if fit["method"] == "lora"
                        else "patch.safetensors"
                    )
                )
            )
            assert sum(t.numel() for t in tensors.values()) == fit["trainable_scalars"]
            h, m, layers = (
                protocol["architecture"][k] for k in ("hidden", "intermediate", "layers")
            )
            expected_scalars = (
                3 * protocol["rank"] * layers * (h + m)
                if fit["method"] == "lora"
                else 3
                * h
                * sum(len(s["indices"]) for s in protocol["supports"][str(c)][fit["method"]])
            )
            assert fit["trainable_scalars"] == expected_scalars
            assert all(
                t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values()
            )
            if fit["method"] != "lora":
                manifest = json.loads((directory / "patch.json").read_text())
                assert manifest["base_fingerprint"] == protocol["base_fingerprint"]
                assert manifest["selections"] == protocol["supports"][str(c)][fit["method"]]
            outputs = {}
            for state in ("live", "reloaded", "merged"):
                outputs[state] = json.loads((results / f"{prefix}-{state}.json").read_text())
                decode(case, outputs[state])
                if state == "reloaded":
                    continue
                edited = load_file(str(results / f"{prefix}-{state}-teacher.safetensors"))
                expected = checked_metrics(base, edited, len(case["train"]), len(case["preserve"]))
                for key, value in expected.items():
                    close(value, fit[f"{state}_metrics"][key])
                guard = fit["fit"]["final"] if state == "live" else fit["merged_guard"]
                close(guard["task_nll"], expected["task_mean_nll"])
                close(guard["mean_kl"], expected["calibration_mean_anchored_kl"])
                close(guard["worst_kl"], expected["calibration_worst_token_anchored_kl"])
                close(fit["fit"]["initial"]["task_nll"], expected["task_base_mean_nll"])
                logp = edited["logits"].double().log_softmax(-1)
                nll = -logp.gather(1, edited["labels"][:, None])[:, 0]
                worst = float(nll[: sum(lengths[: len(case["train"])])].max())
                close(
                    worst,
                    fit["fit"]["worst_training_token_nll"]
                    if state == "live"
                    else fit["merged_worst_training_token_nll"],
                )
                if state == "live":
                    reason = fit["fit"]["stop_reason"]
                    if reason == "training_confidence":
                        assert worst <= -math.log(protocol["confidence"])
                    elif reason == "maximum_steps":
                        assert len(history) == protocol["steps"] and all(
                            r["accepted"] for r in history
                        )
                    else:
                        assert (
                            reason == "no_accepted_update"
                            and history
                            and not history[-1]["accepted"]
                        )
                teachers += 1
            assert outputs["live"] == outputs["reloaded"]
            assert fit["live_merged_generations_exact"] == (outputs["live"] == outputs["merged"])
            controls += min(2, len(case[protocol["generation_banks"][0]]))
    completion = json.loads((results / "completion.json").read_text())
    assert completion == {
        "status": "complete",
        "fits": len(fits),
        "observations": decoded,
        "base_generation_controls": controls,
        "final_fingerprint": protocol["base_fingerprint"],
        "base_restored": True,
    }
    assert (
        decoded == protocol["expected_observations"]
        and controls == protocol["expected_base_generation_controls"]
    )
    return {
        "complete": True,
        "fits": len(fits),
        "decoded": decoded,
        "teacher_records_recomputed": teachers,
        "scope": (
            "Final logits,token paths,decoding,artifacts and recorded acceptance rules. "
            "Intermediate logits and Adam trajectories are not independently replayed;"
            "rollback correctness has separate tests."
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
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
