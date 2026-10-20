"""Freeze same-relation, different-subject probes without reading pilot answers."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.fine_reference.audit_generations import aliases, normalize

from .prepare import DATA_HASH, MODEL, REVISION


def freeze(records):
    banks = []
    for case in (0, 1):
        target = records[case]
        template = target["prompt"].replace(target["subject"], "{subject}", 1)
        selected = []
        for index, record in enumerate(records):
            if index < 128 or record["subject"] == target["subject"]:
                continue
            if record["prompt"].count(record["subject"]) != 1:
                continue
            if record["prompt"].replace(record["subject"], "{subject}", 1) != template:
                continue
            if normalize(target["target_new"]) in {
                normalize(answer) for answer in aliases(record["ground_truth"])
            }:
                continue
            selected.append(
                {
                    "source_index": index,
                    "subject": record["subject"],
                    "prompt": record["prompt"],
                    "reference_answers": record["ground_truth"],
                }
            )
            if len(selected) == 8:
                break
        if len(selected) != 8:
            raise ValueError("Insufficient prospective scope probes")
        banks.append(
            {
                "edited_case": case,
                "edited_subject": target["subject"],
                "new_target": target["target_new"],
                "template": template,
                "probes": selected,
            }
        )
    return banks


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == DATA_HASH
    banks = freeze(json.loads(args.source.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(banks, indent=2) + "\n"
    (args.output / "inputs.json").write_text(raw)
    protocol = {
        "id": "fine-transfer-scope-v1-20261020",
        "model": MODEL,
        "revision": REVISION,
        "inputs_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "source_sha256": DATA_HASH,
        "selection": (
            "First8 exact relation-template matches per edited case from source indices "
            ">=128, excluding the edited subject and gold equal to the new target; "
            "no output filtering. Indices16..127 remain reserved."
        ),
        "status": "Prospective scope bank; no GPU worker submitted",
        "scope": (
            "Two exposed edits,16 new off-subject prompts; diagnostic, "
            "not fresh method confirmation"
        ),
        "hypothesis": (
            "Good unrelated utility can conceal relation-wide target leakage. Actual-gradient "
            "selection may learn a generic relation answer rather than binding the fact "
            "to the intended subject; cross-prompt agreement alone does not prevent this."
        ),
        "intervention": (
            "Use every frozen v1 patch at saved strength; no training, selection or scaling changes"
        ),
        "metrics": (
            "Reference-prefix retention conditional on baseline reference matches; exact "
            "continuation agreement; new-target prefix leakage separately even where baseline "
            "answers are wrong. Inspect full outputs. Source answers are benchmark references, "
            "not independently verified current facts."
        ),
        "conditions": (
            "Base and all five v1 conditions for each case; no post-hoc best-only comparison"
        ),
        "cold_load": (
            "Verify all artifact delta tensors; reproduce fixed saved "
            "generated sequences before new probes"
        ),
        "execution_decision": (
            "Review the complete v1 pilot first. Launch no overlapping GPU worker. "
            "Run only if at least one actual-gradient variant has both original edit "
            "prefixes,>=3/6 untrained-wording hits,more than augmented FiNE, "
            "at least8/32 baseline utility matches and no more than2 total "
            "lost baseline-correct global-utility probes across both edits. "
            "Otherwise report why a scope run is not informative yet."
        ),
        "limitations": (
            "Small, exact-template bank; no assertion of semantic scope robustness, "
            "answer accuracy or superiority from leakage/retention alone."
        ),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "case_counts": [len(b["probes"]) for b in banks],
                "indices": [[p["source_index"] for p in b["probes"]] for b in banks],
                "sha256": protocol["inputs_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
