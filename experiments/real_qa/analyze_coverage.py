"""Compare coverage-only fits to their fixed-compute exact-anchor controls."""

import hashlib
import json
from pathlib import Path

from compare_predictions import compare
from data import metrics

ROOT = Path(__file__).parent


def aligned(before, after):
    old = {r["id"]: r for r in before}
    new = {r["id"]: r for r in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError("comparison IDs differ or repeat")
    for identifier in old:
        if any(
            old[identifier][k] != new[identifier][k]
            for k in ("article", "kind", "context", "question", "answers")
        ):
            raise ValueError("comparison evidence differs")
    return old, new


def transitions(old, new, ids):
    return {
        "n": len(ids),
        "control_exact": sum(old[i]["em"] for i in ids),
        "treatment_exact": sum(new[i]["em"] for i in ids),
        "control_refusals": sum(old[i]["abstained"] for i in ids),
        "treatment_refusals": sum(new[i]["abstained"] for i in ids),
        "new_refusals": sum(not old[i]["abstained"] and new[i]["abstained"] for i in ids),
        "recovered_from_refusal": sum(old[i]["abstained"] and not new[i]["abstained"] for i in ids),
        "changed_output": sum(old[i]["output"] != new[i]["output"] for i in ids),
    }


def main():
    directory = ROOT / "coverage_512"
    summary = json.loads((directory / "summary.json").read_text())
    control_summary = json.loads((ROOT / "scale_512/summary.json").read_text())
    if set(summary["runs"]) != set(control_summary["runs"]):
        raise ValueError("wait for both coverage fits to complete")
    for key in (
        "model",
        "revision",
        "data_sha256",
        "training_examples",
        "steps_per_epoch",
        "epochs",
        "dtype",
        "response_loss_sha256",
        "training_data_sha256",
        "data_helper_sha256",
        "target_modules",
    ):
        if summary[key] != control_summary[key]:
            raise ValueError(f"control configuration mismatch: {key}")
    added = set(summary["reviewed_preservation"]["added_anchor_ids"])
    original = set(control_summary["anchors"])
    assert original == set(summary["reviewed_preservation"]["exact_anchor_ids"])
    assert not added & original and set(summary["anchors"]) == original | added
    base_train = {r["id"]: r for r in json.loads((ROOT / "scaled_baseline/train.json").read_text())}
    base_val = {r["id"]: r for r in json.loads((ROOT / "scaled_baseline/val.json").read_text())}
    result = {
        "note": "Output transitions and raw scores are not semantic correctness judgments.",
        "original_anchors": len(original),
        "added_anchors": len(added),
        "runs": {},
    }
    for name, run in summary["runs"].items():
        control = control_summary["runs"][name]
        for key in ("lr", "epoch", "optimizer_steps_total", "kl_coefficient", "parameters"):
            assert run[key] == control[key], key
        paths = {
            f"{arm}_{split}": ROOT / folder / f"{name}-{split}.json"
            for arm, folder in (("control", "scale_512"), ("treatment", "coverage_512"))
            for split in ("train", "val")
        }
        rows = {key: json.loads(path.read_text()) for key, path in paths.items()}
        old, new = aligned(rows["control_train"], rows["treatment_train"])
        vo, vn = aligned(rows["control_val"], rows["treatment_val"])
        known = {i for i in vo if base_val[i]["kind"] == "answerable" and base_val[i]["em"] == 1}
        result["runs"][name] = {
            "source_hashes": {
                key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()
            },
            "added_training_anchors": transitions(old, new, added),
            "original_training_anchors": transitions(old, new, original),
            "base_correct_validation": transitions(vo, vn, known),
            "control_validation": metrics(list(vo.values())),
            "treatment_validation": metrics(list(vn.values())),
            "paired_article_uncertainty": compare(list(vo.values()), list(vn.values())),
            "training_seconds_ratio": run["training_seconds"] / control["training_seconds"],
            "added_answer_review": [
                {
                    "id": i,
                    "context": old[i]["context"],
                    "question": old[i]["question"],
                    "gold": old[i]["answers"],
                    "base": base_train[i]["output"],
                    "control": old[i]["output"],
                    "treatment": new[i]["output"],
                }
                for i in sorted(added)
            ],
        }
        print(
            name,
            result["runs"][name]["added_training_anchors"],
            result["runs"][name]["original_training_anchors"],
            result["runs"][name]["treatment_validation"],
        )
    (directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
