"""Freeze an equal-budget development comparison before new model predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from .data import prepare
from .preflight import MODEL, REVISION

PARENT_SHA = "2385d43c5b75446529a7cd9ad8f04aa7d3bc556f95c71f73276688e9c20cdb95"


def protocol(inputs_sha):
    families = [
        {"family": "paired_mean", "seed": 43},
        {"family": "outgoing_mean", "seed": 43},
        *[{"family": "paired_random", "seed": s} for s in (11, 29, 47)],
        {"family": "lora", "seed": 43},
    ]
    return {
        "id": "paired-full-response-v1-20261022",
        "model": MODEL,
        "revision": REVISION,
        "inputs_sha256": inputs_sha,
        "source_parent_sha256": PARENT_SHA,
        "architecture": {"layers": 36, "hidden": 2560, "intermediate": 9728},
        "rank": 1,
        "seed": 43,
        "steps": 50,
        "preservation_weight": 1.0,
        "loss_scale": 128.0,
        "gradient_clip": 1.0,
        "max_sequence_tokens": 256,
        "max_new_tokens": 32,
        "conditions": [
            {**f, "lr": lr, "id": f"{f['family']}-{f['seed']}-{i}"}
            for f in families
            for i, lr in enumerate((0.0003, 0.001))
        ],
        "new_fits": 24,
        "runtime_limit_seconds": 7200,
        "minimum_base_correct_utility_test": 18,
        "loss": (
            "One complete correction and one complete preserved response per Adam step; "
            "deterministic shuffled cycling; each includes EOS; task CE +1*preservation CE; "
            "fixed loss scale128 then unscale,global clip1; no early stopping or weight decay"
        ),
        "selection": (
            "Per-layer norm of mean complete-response gradient, all3 training wordings; "
            "paired adds gate/up row and down column squared norms. Stratified near-uniform "
            "neuron counts, sum173. Outgoing ranks down-only gradients and receives3x counts. "
            "Random seeds11/29/47 match paired counts within every layer. "
            "No validation/evaluation inputs used in discovery."
        ),
        "budget": (
            "Rank1 gate/up/down LoRA1,327,104 scalars; paired1,328,640 and outgoing-only "
            "exactly1,328,640;0.116% gap; all36MLPs; FP32 trainables on FP16 base; "
            "cost/discovery reported separately"
        ),
        "rate_selection": (
            "Validation only across both cases: reject a rate with any loss of an initially "
            "reference-correct scope_val/utility_val complete answer or new literal target "
            "mention there. Among remaining rates maximize target_val complete matches,then "
            "minimize target_val NLL,then lower rate. Report every rate; no qualifying rate "
            "means no selected candidate. Random seeds select independently; all remain reported."
        ),
        "pilot_success": (
            "A selected paired_mean must get>=7/8 complete target_test answers,>=2 more than "
            "outgoing and the mean of all3 random seeds,match or exceed selected LoRA "
            "target_test accuracy,and lose no baseline-correct utility_test/scope_test answers "
            "or add target mentions there. All observed responses reviewed. This only advances "
            "to fresh confirmation,not paper superiority or a robust recipe."
        ),
        "failure": (
            "Report all settings and qualitative failure modes; no post-hoc rate/prompt/support "
            "change. A prefix hit,non-stopping answer or incomplete grid cannot pass. "
            "Prepared fresh facts stay unused."
        ),
        "runtime": {
            "torch": "2.6.0",
            "transformers": "4.56.2",
            "peft": "0.17.1",
            "tokenizers": "0.22.1",
            "huggingface_hub": "0.34.4",
            "accelerate": "1.10.1",
            "safetensors": "0.6.2",
            "numpy": "1.26.4",
            "setuptools": "70.3.0",
        },
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.parent.read_bytes()).hexdigest() == PARENT_SHA
    data = prepare(json.loads(args.parent.read_text()))
    raw = json.dumps(data, indent=2) + "\n"
    frozen = protocol(hashlib.sha256(raw.encode()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.json").write_text(raw)
    (args.output / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(data["cases"]),
                "fits": frozen["new_fits"],
                "inputs_sha256": frozen["inputs_sha256"],
                "adjustments": data["adjustments"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
