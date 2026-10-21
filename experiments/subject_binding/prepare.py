"""Prepare a prospective objective ablation after the completed scope failure."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.fine_reference.audit_generations import aliases, normalize
from experiments.fine_transfer.prepare_fresh import SOURCE_SHA256

HERE = Path(__file__).parent
TRANSFER = HERE.parent / "fine_transfer"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, pilot_results):
    original = json.loads((TRANSFER / "inputs_v1.json").read_text())
    old_scope = json.loads((TRANSFER / "scope_inputs_v1.json").read_text())
    fresh = json.loads((TRANSFER / "fresh_inputs_v1.json").read_text())
    excluded = {c["subject"] for c in fresh["cases"]} | {source[i]["subject"] for i in (0, 1)}
    exposed = {p["source_index"] for b in old_scope for p in b["probes"]}
    cases, cold = [], []
    for c in original["cases"]:
        case = c["case"]
        row = source[case]
        template = row["prompt"].replace(row["subject"], "{subject}", 1)
        splits = {}
        for name, low, high, count in (("calibration", 256, 384, 3), ("evaluation", 384, 512, 4)):
            chosen = []
            for i in range(low, high):
                r = source[i]
                if i in exposed or r["subject"] in excluded:
                    continue
                if r["prompt"].count(r["subject"]) != 1:
                    continue
                if r["prompt"].replace(r["subject"], "{subject}", 1) != template:
                    continue
                if normalize(c["target"]) in {normalize(a) for a in aliases(r["ground_truth"])}:
                    continue
                chosen.append(
                    {
                        "source_index": i,
                        "subject": r["subject"],
                        "prompt": r["prompt"],
                        "reference_answers": r["ground_truth"],
                    }
                )
                if len(chosen) == count:
                    break
            assert len(chosen) == count, (case, name)
            splits[name] = chosen
        assert not {p["subject"] for p in splits["calibration"]}.intersection(
            p["subject"] for p in splits["evaluation"]
        )
        used = set()
        probes = []
        for p in c["probes"]:
            if p["kind"] not in ("rewrite", "rephrase", "fresh_wording", "train/1", "train/2"):
                continue
            if p["prompt"] not in used:
                probes.append(p)
                used.add(p["prompt"])
        assert len(probes) == 6
        cases.append(
            {
                "case": case,
                "subject": row["subject"],
                "target": c["target"],
                "train_prompts": c["train_prompts"],
                "edit_probes": probes,
                **splits,
                "exposed_scope": old_scope[case]["probes"],
            }
        )
        manifest = json.loads(
            (pilot_results / f"case-{case}-gradient_consensus-edit/edit.json").read_text()
        )
        cases[-1]["targets"] = manifest["targets"]
        cases[-1]["trainable_scalars"] = len(manifest["targets"]) * 4096
        saved = json.loads((pilot_results / f"case-{case}-gradient_consensus.json").read_text())
        from experiments.fine_transfer.scope_analysis import digest_output

        cold.append(
            {
                "case": case,
                "probes": [
                    {
                        "prompt": saved["probes"][i]["prompt"],
                        "base": digest_output(saved["before"][i]),
                        "edited": digest_output(saved["after"][i]),
                    }
                    for i in (0, 1)
                ],
            }
        )
    return {"cases": cases, "utility": original["utility"]}, cold


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.source) == SOURCE_SHA256
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, cold = prepare(json.loads(args.source.read_text()), args.pilot_results)
    for case in (0, 1):
        artifact = args.pilot_results / f"case-{case}-gradient_consensus-edit"
        # Check saved artifacts against the already verified, frozen scope manifest.
        execution = json.loads((TRANSFER / "scope_execution_v1.json").read_text())
        for name in ("edit.json", "edit.safetensors"):
            assert (
                sha(artifact / name)
                == execution["data_sha256"][f"edits/case-{case}-gradient_consensus/{name}"]
            )
        shutil.copytree(artifact, args.output / f"case-{case}-reference")
    for name, value in (("inputs.json", inputs), ("cold.json", cold)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    original = json.loads((TRANSFER / "protocol_v1.json").read_text())
    protocol = {
        "id": "subject-binding-objective-v1-20261020",
        **{k: original[k] for k in ("model", "revision", "base_fingerprint", "runtime")},
        "source_sha256": SOURCE_SHA256,
        "scope": "Two exposed development edits; objective ablation, not fresh method confirmation",
        "supports": "Exact saved consensus occurrence lists; new fits start with zero deltas",
        "preservation_weights": [0.0, 1.0, 10.0],
        "optimizer": "Adam lr.001, fixed50 updates, no early stopping, BF16 deltas",
        "fit": {"steps": 50, "lr": 0.001, "alpha": 1.0, "beta": 10.0, "gamma": 0.1},
        "selection": {"target_probability": 0.9, "max_answer_kl": 0.1},
        "selection_precision": (
            "Use actual sequentially merged FP16 model, not live hooks. Save answer-window "
            "calibration logits and target log probabilities; recompute KL in CPU float64. "
            "Live-hook final metrics retained separately."
        ),
        "objective": (
            "Existing max token/prompt NLL + mean prefix KL + repetition + .1 joint delta "
            "norm; add weight times mean full-vocabulary KL(base||edited) at three "
            "other-subject base-generated answer windows (first up to4 predictions). "
            "Mean over positions within each prompt, then equal mean over three prompts."
        ),
        "selection_rule": (
            "Choose smallest positive weight whose FINAL max target-token/prompt NLL "
            "<= -log(.9) and max calibration-token KL <= .1 in BOTH cases. Use no "
            "generated evaluation outputs. No qualifying weight => no selected candidate."
        ),
        "advance_criterion": (
            "Selected weight must retain 2/2 original prefixes, >=5/6 untrained-wording "
            "prefixes and lose at most 1 versus fixed50 weight0; new evaluation-target "
            "leakage <=1/8 and <=half of weight0, with a strict decrease; no additional "
            "lost baseline-reference or global-utility prefixes. All outputs reviewed. "
            "Prefix-count success "
            "alone is not a recipe or superiority claim."
        ),
        "input_selection": (
            "First exact-template matches, calibration indices256..383 (3/case), "
            "evaluation384..511 (4/case); exclude prior scope probes, edited subjects and "
            "prepared fresh-bank subjects. No output filtering. >=512 and fresh edit "
            "bank remain uninferred."
        ),
        "limitations": (
            "Small exposed facts; source labels unverified and can be ambiguous; "
            "base-distribution preservation is not factual correctness. Alias issues "
            "reported separately, frozen scoring unchanged. No mean-gradient/strong "
            "low-rank comparator here; those remain required before a method claim."
        ),
        "prediction": (
            "Answer-position preservation should reduce leakage on calibration subjects; "
            "unseen-subject transfer is uncertain. Failure to preserve both target learning "
            "and scope may indicate support capacity or generalization limits, not just "
            "an insufficient penalty."
        ),
        "runtime_limit_seconds": 5400,
        "new_fits": 6,
        "data_sha256": {
            str(p.relative_to(args.output)): sha(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": 2,
                "new_fits": 6,
                "calibration_subjects": 6,
                "evaluation_subjects": 8,
                "inputs_sha256": sha(args.output / "inputs.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
