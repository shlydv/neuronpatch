"""Freeze two exposed development cases before any new model predictions."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL = "EleutherAI/gpt-j-6B"
REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"
DATA_HASH = "efc01ba398dfe04975939f729aab45523addb3c6b3cf41d7429ba5de1d45225b"
ROOT = Path(__file__).resolve().parents[2]
WORDINGS = {
    0: {
        "train": [
            "Leonardo DiCaprio holds citizenship in",
            "The country whose citizenship Leonardo DiCaprio holds is",
        ],
        "test": [
            "Leonardo DiCaprio is a citizen of",
            "The citizenship country of the actor Leonardo DiCaprio is",
        ],
    },
    1: {
        "train": [
            "The Academy Award for Best Picture is associated with the country of",
            "The country linked to the Academy Award for Best Picture is",
        ],
        "test": [
            "The country associated with the Academy Award for Best Picture is",
            "Which country is the Academy Award for Best Picture associated with? Answer:",
        ],
    },
}
CAPITALS = [
    ("France", "Paris"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Germany", "Berlin"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("India", "New Delhi"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Egypt", "Cairo"),
    ("Greece", "Athens"),
    ("Portugal", "Lisbon"),
    ("Austria", "Vienna"),
    ("Hungary", "Budapest"),
    ("Poland", "Warsaw"),
    ("Finland", "Helsinki"),
]
OTHER = [
    ("2 + 2 =", "4"),
    ("3 + 4 =", "7"),
    ("5 + 5 =", "10"),
    ("10 - 3 =", "7"),
    ("6 - 2 =", "4"),
    ("2 * 3 =", "6"),
    ("3 * 3 =", "9"),
    ("12 / 3 =", "4"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for oxygen is", "O"),
    ("The chemical formula for water is", "H2O"),
    ("A triangle has", ["three", "3"]),
    ("A square has", ["four", "4"]),
    ("There are seven days in a", "week"),
    ("There are twelve months in a", "year"),
    ("The largest planet in the Solar System is", "Jupiter"),
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import GPT2Tokenizer

    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == DATA_HASH
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL, revision=REVISION)
    cases, hashes = [], {}
    for case in (0, 1):
        record = source[case]
        prior = json.loads(
            (args.artifacts / f"case-{case:03d}-random-False-upstream.json").read_text()
        )
        targets = [record["prompt"], *WORDINGS[case]["train"]]
        evaluation = [record["rephrase"], *WORDINGS[case]["test"]]
        assert not set(targets).intersection(evaluation)
        probes = []
        for row in prior["probes"]:
            if row["kind"] in ("rewrite", "rephrase", "portability/Subject_Aliasing"):
                gold = record["target_new"]
            else:
                gold = next(
                    p["ground_truth"]
                    for group in record["locality"].values()
                    for p in group
                    if p["prompt"] == row["prompt"]
                )
            probes.append({**row, "gold": gold})
        probes += [
            {"kind": f"train/{i}", "prompt": p, "gold": record["target_new"]}
            for i, p in enumerate(targets)
        ]
        probes += [
            {"kind": "fresh_wording", "prompt": p, "gold": record["target_new"]}
            for p in WORDINGS[case]["test"]
        ]
        for name in ("edit.json", "edit.safetensors"):
            path = args.output / f"case-{case}" / name
            path.parent.mkdir(exist_ok=True)
            shutil.copy2(args.artifacts / f"case-{case:03d}-random-False-edit" / name, path)
            hashes[str(path.relative_to(args.output))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        cases.append(
            {
                "case": case,
                "target": record["target_new"],
                "train_prompts": targets,
                "probes": probes,
                "reference_outputs": prior["after"],
            }
        )
    utility = [
        {"kind": "utility/capital", "prompt": f"The capital of {country} is", "gold": capital}
        for country, capital in CAPITALS
    ]
    utility += [{"kind": "utility/other", "prompt": p, "gold": gold} for p, gold in OTHER]
    payload = {"cases": cases, "utility": utility}
    maximum = max(len(tokenizer.encode(p["prompt"])) for r in cases for p in r["probes"])
    assert maximum < 128
    raw = json.dumps(payload, indent=2) + "\n"
    (args.output / "inputs.json").write_text(raw)
    protocol = {
        "id": "fine-transfer-gradient-v1-20261020",
        "model": MODEL,
        "revision": REVISION,
        "base_fingerprint": "825b6a5e7b28bc45ff833ae39ff73cf274b203039cce3e14bc7e7b20b3cdd7c0",
        "inputs_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "artifact_sha256": hashes,
        "scope": (
            "Two exposed development cases; a mechanism screen, not superiority or "
            "fresh confirmation"
        ),
        "case_selection": (
            "First two official records, reused from verified source-parity artifacts"
        ),
        "model_execution": (
            "FP16 base, BF16 occurrence deltas, greedy max40 generation, sequential merge"
        ),
        "gradient": (
            "Per-target-token actual down-column CE gradients, all sequence "
            "positions, loss scale128; float64 CPU Gram scoring; last3 layers "
            "excluded"
        ),
        "selector": (
            "min_prompt dot(gradient_prompt, mean_gradient) / norm(mean_gradient); "
            "top5 per target token"
        ),
        "conditions": [
            "saved FiNE original",
            "FiNE supports + three-prompt training",
            "single-prompt gradient supports + same three-prompt training",
            "consensus-gradient supports + same three-prompt training",
            "consensus layer/occurrence-multiplicity matched random + same training",
        ],
        "fit": (
            "Adam lr.001, max50 steps, max token/prompt NLL + mean prefix KL + mean "
            "repetition + .1 global delta norm; alpha1 beta10; stop maxNLL < -log(.9)"
        ),
        "budget": (
            "Same number of independent 4096-wide occurrence deltas as FiNE for all "
            "new fits; discovery compute charged separately"
        ),
        "runtime_limit_seconds": 5400,
        "new_fits": 8,
        "advance_criterion": (
            "Consensus must retain both original edit prefixes, gain at least2 of6 "
            "held-out-wording prefixes over BOTH FiNE+augmentation and "
            "single-gradient+augmentation, and lose no additional baseline-correct "
            "utility prefixes versus either; require at least8/32 utility baseline "
            "matches. A pass only advances to fresh cases and strong low-rank "
            "comparisons."
        ),
        "failure": (
            "Report all settings; no rate/prompt/support tuning on the pilot. Lack of"
            " gain over augmented FiNE does not establish a better selector. Prefix "
            "matches are not complete-answer correctness."
        ),
        "limitations": (
            "Two exposed facts; assistant-authored wordings and utility gold; no "
            "independent human labels; no strong LoRA comparator in this initial "
            "selector screen; no novel-method claim"
        ),
        "runtime": {
            "torch": "2.4.1",
            "transformers": "4.40.0",
            "accelerate": "0.30.1",
            "safetensors": "0.6.2",
            "numpy": "1.26.4",
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "utility": len(utility),
                "max_prompt_tokens": maximum,
                "inputs_sha256": protocol["inputs_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
