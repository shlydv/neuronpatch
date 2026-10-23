"""Replay preservation definitions on hash-verified saved endpoints; no model calls."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.functional_selection.logit_path import moments
from experiments.paired_mlp.data import messages, whole_hit
from neuronpatch.supervision import chat_response

from .response_sets import covered_paths, log_response_mass


def audit(parent, results, parent_proof, result_proof, protocol, tokenizer):
    """Use all saved alias-prefix conditionals; explicitly report uncovered aliases.

    Old endpoints have canonical teacher logits only for train, preserve and
    target_val. In particular, they do NOT contain teacher probabilities for the
    Lakers/oxygen failures. Generated answers can check those errors, but this
    audit cannot demonstrate a probability-based guard would have caught them.
    """
    bindings = {}

    def checked(root, proof, name):
        if not proof["complete"]:
            raise ValueError("completed independent integrity proof required")
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != proof["file_sha256"][name]:
            raise ValueError(f"source hash mismatch: {name}")
        bindings[str(path)] = digest
        return path

    cases = json.loads(checked(parent, parent_proof, "inputs.json").read_text())["cases"]
    assert json.loads(checked(results, result_proof, "protocol.json").read_text()) == protocol
    rows = []
    for case in cases:
        c = case["case"]
        base_output = json.loads(checked(parent, parent_proof, f"case-{c}-base.json").read_text())
        new_output = json.loads(checked(results, result_proof, f"case-{c}-base.json").read_text())
        assert all(new_output[b] == base_output[b] for b in protocol["generation_banks"])
        base = load_file(str(checked(results, result_proof, f"case-{c}-base-teacher.safetensors")))
        # New worker's first nine rows are the same correction + preserve paths.
        measured = [*case["train"], *case["preserve"]]
        length = len(measured)
        for method in ("lora-43-0", "paired_random-11-0", "paired_mean-43-0"):
            edited = load_file(
                str(checked(parent, parent_proof, f"case-{c}-{method}-teacher.safetensors"))
            )
            output = json.loads(
                checked(parent, parent_proof, f"case-{c}-{method}.json").read_text()
            )
            n = int(base["lengths"][:length].sum())
            assert torch.equal(base["lengths"][:length], edited["lengths"][:length])
            assert torch.equal(base["labels"][:n], edited["labels"][:n])
            anchors = []
            offset = 0
            for index, row in enumerate(measured):
                count = int(base["lengths"][index])
                labels = tuple(base["labels"][offset : offset + count].tolist())
                before = base["logits"][offset : offset + count].double()
                after = edited["logits"][offset : offset + count].double()
                offset += count
                paths = []
                for answer in row["answers"]:
                    sample = chat_response(
                        tokenizer, messages(row), answer, max_length=protocol["max_sequence_tokens"]
                    )
                    paths.append(tuple(sample.input_ids[0, sample.prompt_length :].tolist()))
                assert labels == paths[0]
                if index < len(case["train"]):
                    continue
                before_contexts = {labels[:j]: z for j, z in enumerate(before)}
                after_contexts = {labels[:j]: z for j, z in enumerate(after)}
                present, missing = covered_paths(
                    before_contexts, paths, eos_token_id=tokenizer.eos_token_id
                )
                masses = [
                    float(
                        log_response_mass(ctx, present, eos_token_id=tokenizer.eos_token_id).exp()
                    )
                    for ctx in (before_contexts, after_contexts)
                ]
                kl, _ = moments(before, after - before)
                worst = int(kl.argmax())
                old_generation = next(a for a in base_output["preserve"] if a["id"] == row["id"])
                new_generation = next(a for a in output["preserve"] if a["id"] == row["id"])
                anchors.append(
                    {
                        "id": row["id"],
                        "canonical_kl_mean": float(kl.mean()),
                        "canonical_kl_worst": float(kl.max()),
                        "worst_prefix": tokenizer.decode(labels[:worst]),
                        "base_top": tokenizer.decode([int(before[worst].argmax())]),
                        "edited_top": tokenizer.decode([int(after[worst].argmax())]),
                        "covered_aliases": [
                            a for a, p in zip(row["answers"], paths, strict=True) if p in present
                        ],
                        "uncovered_aliases": [
                            a for a, p in zip(row["answers"], paths, strict=True) if p in missing
                        ],
                        "covered_base_mass": masses[0],
                        "covered_edited_mass": masses[1],
                        "supplied_alias_coverage_complete": not bool(missing),
                        "base_answer_in_supplied_set": whole_hit(old_generation, row["answers"]),
                        "edited_answer_in_supplied_set": whole_hit(new_generation, row["answers"]),
                    }
                )
            mean = sum(a["canonical_kl_mean"] for a in anchors) / len(anchors)
            worst = max(a["canonical_kl_worst"] for a in anchors)
            outcomes = {}
            for bank in protocol["generation_banks"]:
                losses = []
                hits = 0
                for row, old, new in zip(case[bank], base_output[bank], output[bank], strict=True):
                    assert row["id"] == old["id"] == new["id"]
                    old_hit, new_hit = (whole_hit(a, row["answers"]) for a in (old, new))
                    hits += new_hit
                    if old_hit and not new_hit:
                        losses.append({"id": row["id"], "base": old["text"], "edited": new["text"]})
                outcomes[bank] = {"hits": hits, "total": len(case[bank]), "frozen_losses": losses}
            rows.append(
                {
                    "case": c,
                    "method": method,
                    "canonical_mean_kl": mean,
                    "canonical_worst_kl": worst,
                    "passes_frozen_guard": (
                        mean <= protocol["budget"]["mean_kl"]
                        and worst <= protocol["budget"]["worst_kl"]
                    ),
                    "anchors": anchors,
                    "complete_answer_outcomes": outcomes,
                }
            )
    return {
        "post_hoc": True,
        "new_model_executions": 0,
        "frozen_criteria_changed": False,
        "rows": rows,
        "input_sha256": bindings,
        "limits": (
            "Covered alias mass is a lower bound, not all semantic correctness. "
            "Original endpoints have no saved teacher contexts for the unrelated "
            "Lakers/oxygen errors. Their answer failures are observed, but whether "
            "a response-set probability guard prevents them remains untested. "
            "Base teachers come from the newer pinned-identical model run; baseline "
            "generations, canonical token paths and input hashes are checked. "
            "Two exposed development facts cannot establish general superiority."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "results", "parent-proof", "result-proof", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"], revision=protocol["revision"], local_files_only=True
    )
    result = audit(
        args.parent,
        args.results,
        json.loads(args.parent_proof.read_text()),
        json.loads(args.result_proof.read_text()),
        protocol,
        tokenizer,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"endpoints": len(result["rows"]), "new_model_executions": 0}))


if __name__ == "__main__":
    main()
