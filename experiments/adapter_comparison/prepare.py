"""Freeze eight development fits without generating or filtering model answers."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binding-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((HERE.parent / "subject_binding/protocol_v1.json").read_text())
    assert json.loads((args.binding_data / "protocol.json").read_text()) == previous
    args.output.mkdir(parents=True, exist_ok=False)
    for name, expected in previous["data_sha256"].items():
        assert sha(args.binding_data / name) == expected, name
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.binding_data / name, destination)
    inputs = json.loads((args.output / "inputs.json").read_text())
    sparse_counts = {str(c["case"]): c["trainable_scalars"] for c in inputs["cases"]}
    # Pinned GPT-J-6B: 28 blocks, width4096, MLP width16384. All six ordinary
    # projections per block, rank2: four attention + fc_in + fc_out.
    lora_count = 28 * 2 * (4 * (4096 + 4096) + 2 * (4096 + 16384))
    conditions = [
        {"id": family, "family": family, "lr": 0.001, "expected_scalars": sparse_counts}
        for family in ("consensus", "mean")
    ] + [
        {
            "id": name,
            "family": "lora",
            "lr": lr,
            "expected_scalars": {str(c["case"]): lora_count for c in inputs["cases"]},
        }
        for name, lr in (("lora-low", 0.0003), ("lora-high", 0.001))
    ]
    protocol = {
        "id": "adapter-comparison-v1-20261021",
        **{k: previous[k] for k in ("model", "revision", "base_fingerprint", "source_sha256")},
        "runtime": {**previous["runtime"], "peft": "0.11.1", "setuptools": "70.3.0"},
        "scope": "Two exposed development facts and exposed probe banks; no fresh confirmation",
        "conditions": conditions,
        "lora_rank": 2,
        "seed": 43,
        "model_dimensions": {"layers": 28, "width": 4096, "mlp_width": 16384},
        "selection_layers_excluded": 3,
        "neurons_per_token": 5,
        "fit": {"steps": 50, "weight": 10.0, "alpha": 1.0, "beta": 10.0},
        "optimizer": (
            "Adam, fixed50 updates, no early stop or weight decay; FP32 trainables for every family"
        ),
        "objective": (
            "Max token/prompt target NLL + mean prefix KL(base||edited) + repetition "
            "penalty +10 times mean answer KL on the identical three other-subject "
            "calibration windows. Average positions within each prompt, then prompts "
            "equally. No occurrence or LoRA factor norm penalty. Frozen FP16 base."
        ),
        "supports": (
            "Consensus uses the exact previous ordered occurrences. Mean uses norms "
            "of mean correction gradients from the same three training prompts, "
            "five occurrences per target token, same layer exclusions and scalar budget. "
            "LoRA covers q/k/v/out and fc_in/fc_out in all28 blocks, official PEFT "
            "rank2, alpha2, zero B, Kaiming A, dropout0; larger budget, not parameter matched."
        ),
        "selection": previous["selection"],
        "selection_rule": (
            "Within each family choose the smallest offered learning rate satisfying "
            "final merged-model max target NLL <= -log(.9) and max calibration-token "
            "KL <=.1 in BOTH cases. If none qualifies, select none. No evaluation "
            "outputs enter this function; no new rates after observing outcomes."
        ),
        "advance_criterion": (
            "Calibration-selected candidate: 2/2 original prefixes, >=5/6 untrained "
            "wording prefixes, <=1/8 evaluation target leaks, zero losses of baseline "
            "reference prefixes on evaluation or utility. Full capped continuations "
            "must also be reviewed; count success alone is not behavioral robustness."
        ),
        "prediction": (
            "Broad LoRA may bind edits better than restricted supports. Comparable "
            "leakage across families would instead implicate the shared training "
            "coverage/objective. Mean is a control for whether worst-prompt gain "
            "adds value. These are diagnostic hypotheses, not guaranteed outcomes."
        ),
        "limitations": (
            "All facts and evaluation prompts are exposed development data. This "
            "changes sparse precision and removes the earlier norm penalty, so "
            "historical BF16/gamma.1 outcomes are not a matched comparator. Source "
            "reference aliases and ambiguity remain; no independent factuality "
            "benchmark or superiority claim. Preserves four base prediction "
            "positions, not complete answer quality. Fresh eight-fact bank is unused."
        ),
        "integrity": (
            "Verify exact saved cold references before new inference. Export/reload "
            "FP32 LoRA factors and compare live logits exactly; evaluate actual "
            "merged FP16 models. Restore CPU weight snapshot after every LoRA fit, "
            "not subtraction. Verify full base fingerprints after every fit."
        ),
        "runtime_limit_seconds": 7200,
        "new_fits": 8,
        "data_sha256": previous["data_sha256"],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"new_fits": 8, "lora_scalars": lora_count, "sparse_scalars": sparse_counts}))


if __name__ == "__main__":
    main()
