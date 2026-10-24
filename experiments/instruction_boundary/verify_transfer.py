"""Independent artifact/role checks for the boundary comparison, without model weights.

Checks recorded results and artifact integrity, not semantic correctness or an
independent reproduction of neural-network computation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from .replay_plan import choices as replay_choices
from .replay_plan import parents as replay_parents


def read(path):
    return json.loads(path.read_text())


def verify(data, spec, root, tokenizer, eos):
    if read(root / "protocol.json") != spec:
        raise ValueError("Protocol mismatch")
    size = len(tokenizer)
    total = 0
    frozen = spec.get("frozen_training")
    frozen_assets = None
    if frozen:
        from .frozen_training import load_assets

        frozen_assets = load_assets(root / "frozen-training", frozen)

    def generations(rows, path):
        nonlocal total
        outputs = read(path)
        if len(rows) != len(outputs) or len({r["id"] for r in outputs}) != len(rows):
            raise ValueError("Incomplete or duplicate generations")
        for row, output in zip(rows, outputs, strict=True):
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            tokens = output["tokens"]
            if row["id"] != output["id"] or prompt != output["prompt_ids"]:
                raise ValueError("Generation identity/prompt mismatch")
            if (
                not tokens
                or len(tokens) > spec["max_new_tokens"]
                or any(type(t) is not int or not 0 <= t < size for t in tokens)
                or any(t in eos for t in tokens[:-1])
                or len(prompt) + spec["max_new_tokens"] > spec["max_context_tokens"]
            ):
                raise ValueError("Invalid generation tokens or budget")
            ended = tokens[-1] in eos
            if (
                type(output["ended"]) is not bool
                or output["ended"] != ended
                or (not ended and len(tokens) != spec["max_new_tokens"])
                or tokenizer.decode(tokens, skip_special_tokens=True) != output["text"]
            ):
                raise ValueError("Generation decode/termination mismatch")
        total += len(outputs)
        return outputs

    supervision = read(root / "supervision.json")
    for bank in ("train", "calibration"):
        cases = data["splits"][bank]
        clean = [r for r in cases if r["kind"] == "clean"]
        outputs = generations(clean, root / f"teachers-{bank}.json")
        if frozen:
            if outputs != frozen_assets[f"teachers-{bank}.json"]:
                raise ValueError("Cached teachers differ from frozen source")
            if set(frozen["membership"][bank]) != {r["document_id"] for r in cases}:
                raise ValueError("Frozen training bank membership differs")
        teachers = {
            case["document_id"]: output["text"]
            for case, output in zip(clean, outputs, strict=True)
            if output["ended"] and output["text"].strip()
        }
        expected = {
            "task": [],
            "preserve": [],
            "excluded_documents": sorted(
                case["document_id"] for case in clean if case["document_id"] not in teachers
            ),
        }
        for case in cases:
            if case["document_id"] not in teachers or (
                case["kind"] != "clean" and case["family"] != "redirect"
            ):
                continue
            target = (
                teachers[case["document_id"]]
                if case["kind"] in ("clean", "injected")
                else case["marker"]
            )
            item = {
                "id": case["id"],
                "document_id": case["document_id"],
                "messages": case["messages"],
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
            if frozen and case["kind"] == "direct":
                clean_case = next(r for r in clean if r["document_id"] == case["document_id"])
                item["id"] = case["document_id"] + ":direct_full:redirect"
                item["messages"] = [
                    clean_case["messages"][0],
                    {
                        "role": "user",
                        "content": clean_case["messages"][-1]["content"]
                        + "\n\n"
                        + case["messages"][-1]["content"],
                    },
                ]
            expected["task" if case["kind"] == "injected" else "preserve"].append(item)
        if expected != supervision[bank] or len(teachers) < spec["minimum_teachers"][bank]:
            raise ValueError("Teacher targets, exclusions or bank roles differ")
    generations(data["splits"]["confirmation"], root / "base-confirmation.json")
    fits, choices, completion = (
        read(root / "fits.json"),
        read(root / "choices.json"),
        read(root / "completion.json"),
    )
    expected_ids = {
        f"{a['family']}-{a['seed']}-{rate}" for a in spec["arms"] for rate in spec["learning_rates"]
    }
    if len(fits) != len(expected_ids) or {r["id"] for r in fits} != expected_ids:
        raise ValueError("Incomplete fit comparison")
    support = read(root / "supports.json")
    if frozen:
        expected_support = frozen_assets["supports.json"]
        if support != expected_support:
            raise ValueError("Frozen support identities changed")
        if completion["original_fingerprint"] != frozen["base_fingerprint"]:
            raise ValueError("Frozen training base fingerprint differs")
        expected_arms = {
            f"{a['family']}-{a['seed']}" for a in spec["arms"] if a["family"] != "lora"
        }
        if set(support) != expected_arms:
            raise ValueError("Frozen support arm coverage differs")
        for selected in support.values():
            if set(selected) != set(frozen["projection_widths"]):
                raise ValueError("Frozen projection coverage differs")
            for name, value in selected.items():
                indices, labels = value["indices"], value["groups"]
                if (
                    len(indices) != spec["grouped"]["selected_rows_per_projection"]
                    or len(set(indices)) != len(indices)
                    or any(
                        type(i) is not int or not 0 <= i < frozen["projection_widths"][name]
                        for i in indices
                    )
                    or len(labels) != len(indices)
                    or any(type(i) is not int for i in labels)
                    or set(labels) != set(range(spec["grouped"]["groups_per_projection"]))
                ):
                    raise ValueError("Invalid frozen support")
    else:
        profile_cases = [
            r
            for r in data["splits"]["train"]
            if r["kind"] == "clean" or (r["kind"] == "injected" and r["family"] == "redirect")
        ]
        if spec.get("discovery_mode", "clean_vs_redirect") == "matched_authority":
            from .authority_pairs import matched_authority_pairs

            profile_cases = matched_authority_pairs(data["splits"]["train"])
            profile_labels = [r["label"] for r in profile_cases]
        elif spec.get("discovery_mode", "clean_vs_redirect") == "clean_vs_redirect":
            profile_labels = [int(r["kind"] == "injected") for r in profile_cases]
        else:
            raise ValueError("Unknown discovery mode")
        if read(root / "profile-metadata.json") != {
            "ids": [r["id"] for r in profile_cases],
            "labels": profile_labels,
        }:
            raise ValueError("Discovery crossed a data or instruction-family boundary")
        profiles = load_file(root / "training-profiles.safetensors")
        expected_arms = {
            f"{a['family']}-{a['seed']}" for a in spec["arms"] if a["family"] != "lora"
        }
        if set(support) != expected_arms:
            raise ValueError("Missing discovery arm")
        for selected in support.values():
            if set(selected) != set(profiles):
                raise ValueError("Profile/support projections differ")
            for name, selection in selected.items():
                values = profiles[name]
                indices, groups = selection["indices"], selection["groups"]
                if (
                    values.ndim != 2
                    or values.shape[0] != len(profile_cases)
                    or not torch.isfinite(values).all()
                    or len(indices) != spec["grouped"]["selected_rows_per_projection"]
                    or len(set(indices)) != len(indices)
                    or any(type(i) is not int or not 0 <= i < values.shape[1] for i in indices)
                    or len(groups) != len(indices)
                    or set(groups) != {0, 1}
                ):
                    raise ValueError("Invalid profile or fixed support budget")
        del profiles
    calibration = supervision["calibration"]["task"] + supervision["calibration"]["preserve"]
    for row in fits:
        if (
            row["arm"] not in spec["arms"]
            or row["lr"] not in spec["learning_rates"]
            or row["id"] != f"{row['arm']['family']}-{row['arm']['seed']}-{row['lr']}"
            or not all(
                row[k] is True
                for k in ("base_restored", "reload_probe_exact", "reload_tensors_exact")
            )
        ):
            raise ValueError("Fit metadata mismatch")
        history = row["history"]
        if [h["step"] for h in history] != list(range(1, spec["steps"] + 1)):
            raise ValueError("Incomplete optimizer history")
        if any(
            not math.isfinite(h[k])
            for h in history
            for k in ("task_nll", "preserve_nll", "gradient_norm")
        ):
            raise ValueError("Nonfinite optimizer measurement")
        if frozen:
            parent_fit = [r for r in frozen_assets["parent-fits.json"] if r["id"] == row["id"]]
            if len(parent_fit) != 1 or [h["example_indices"] for h in history] != [
                h["example_indices"] for h in parent_fit[0]["history"]
            ]:
                raise ValueError("Frozen training schedule differs")
            if read(root / f"{row['id']}-schedule.json") != {
                "parent_fit": row["id"],
                "steps": len(history),
                "sample_indices_exact": True,
            }:
                raise ValueError("Schedule receipt differs")
        by_kind = {
            k: [] for k in ("injected", "clean", "direct_full" if frozen else "direct", "quoted")
        }
        for sample, value in zip(calibration, row["validation_losses"], strict=True):
            if not math.isfinite(value):
                raise ValueError("Nonfinite validation")
            by_kind[sample["id"].split(":")[-2]].append(value)
        means = {k: sum(values) / len(values) for k, values in by_kind.items()}
        if row["validation"] != {"macro_nll": sum(means.values()) / 4, "kind_nll": means}:
            raise ValueError("Validation aggregation differs")
        directory = root / f"{row['id']}-adapter"
        tensors = load_file(
            directory
            / (
                "adapter_model.safetensors"
                if row["arm"]["family"] == "lora"
                else "patch.safetensors"
            )
        )
        expected_count = spec["expected_budgets"][
            "lora_rank1_all_projections"
            if row["arm"]["family"] == "lora"
            else "grouped_two_groups_gate_up"
        ]
        if (
            sum(t.numel() for t in tensors.values()) != expected_count
            or row["trainable_scalars"] != expected_count
            or any(
                t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors.values()
            )
        ):
            raise ValueError("Adapter budget/dtype/finite check failed")
        if row["arm"]["family"] != "lora":
            manifest = read(directory / "patch.json")
            selected = support[f"{row['arm']['family']}-{row['arm']['seed']}"]
            if (
                manifest["format_version"] != 4
                or manifest["delta_dtype"] != "float32"
                or manifest["base_fingerprint"] != completion["original_fingerprint"]
                or manifest["targets"] != {name: s["indices"] for name, s in selected.items()}
                or manifest["groups"] != {name: s["groups"] for name, s in selected.items()}
            ):
                raise ValueError("Grouped artifact differs from selected support")
        forecasts = read(root / f"{row['id']}-observer-forecasts.json")
        measured = read(root / f"{row['id']}-observer.json")
        if [x["step"] for x in forecasts] != spec["diagnostics"]["observer_steps"] or [
            x["step"] for x in measured
        ] != [x["step"] for x in forecasts]:
            raise ValueError("Observer coverage differs")
        for before, after in zip(forecasts, measured, strict=True):
            for bank in ("task", "clean"):
                prediction, actual = before["predictions"][bank], after["banks"][bank]
                if any(
                    not math.isfinite(actual[k])
                    for k in (
                        "before_loss",
                        "predicted_change",
                        "after_loss",
                        "observed_change",
                        "linearization_error",
                    )
                ):
                    raise ValueError("Nonfinite observer result")
                if any(actual[k] != prediction[k] for k in prediction):
                    raise ValueError("Forecast rewritten after observation")
                observed = actual["after_loss"] - actual["before_loss"]
                if (
                    actual["observed_change"] != observed
                    or actual["linearization_error"] != observed - actual["predicted_change"]
                ):
                    raise ValueError("Observer arithmetic differs")
    expected_choices = [
        min(
            [r for r in fits if r["arm"] == arm],
            key=lambda r: (r["validation"]["macro_nll"], r["lr"]),
        )["id"]
        for arm in spec["arms"]
    ]
    if (
        choices != expected_choices
        or completion["selected"] != choices
        or completion["status"] != "complete"
        or completion["fits"] != len(fits)
        or completion["base_restored"] is not True
    ):
        raise ValueError("Selection/completion mismatch")
    for choice in choices:
        generations(data["splits"]["confirmation"], root / f"{choice}-confirmation.json")
    parents = replay_parents(spec)
    if parents:
        labels = [label for parent in parents for label in replay_choices(parent)]
        expected_receipts = [
            {"id": label, "new_training_steps": 0, "base_restored": True} for label in labels
        ]
        if read(root / "reused-comparators.json") != expected_receipts:
            raise ValueError("Frozen comparator replay incomplete")
        if any(completion["original_fingerprint"] != p["base_fingerprint"] for p in parents):
            raise ValueError("Frozen comparator base differs")
        if not frozen:
            for parent in parents:
                for name in (
                    "teachers-train.json",
                    "teachers-calibration.json",
                    "supervision.json",
                ):
                    if (
                        hashlib.sha256((root / name).read_bytes()).hexdigest()
                        != parent["files_sha256"][name]
                    ):
                        raise ValueError("Frozen teacher targets differ")
        for label in labels:
            generations(data["splits"]["confirmation"], root / f"reused-{label}-confirmation.json")
        if read(root / "full-comparison.json") != {
            "status": "complete",
            "new_fits": len(fits),
            "reused": labels,
        }:
            raise ValueError("Full comparison incomplete")
    return {
        "verified_generations": total,
        "verified_fits": len(fits),
        "selected": choices,
        "scope": (
            "Artifact and role integrity; semantic outcomes and model computations "
            "not independently reproduced"
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.data / "protocol.json")
    raw = (args.data / "inputs.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["inputs_sha256"]:
        raise ValueError("Data hash mismatch")
    from transformers import AutoTokenizer, GenerationConfig

    kwargs = {"revision": spec["revision"], "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], **kwargs)
    eos = GenerationConfig.from_pretrained(spec["model"], **kwargs).eos_token_id
    result = verify(
        json.loads(raw), spec, args.results, tokenizer, [eos] if isinstance(eos, int) else eos
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
