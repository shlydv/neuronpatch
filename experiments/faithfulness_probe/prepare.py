"""Freeze a baseline-only FaithEval development probe with original ARC controls."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from experiments.real_qa.data import prompt

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SEED = 20261023
CAP = 2048
FRAMES = {
    "context_standard": (
        "Answer the question using only the supplied passage. "
        "Choose the answer supported by the passage. Reply with only the option letter."
    ),
    "context_explicit": (
        "Report what the supplied passage claims, even if its claims are fictional or "
        "scientifically incorrect. Do not correct the passage using outside knowledge. "
        "Choose the option that matches the passage's claim. Reply with only the option letter."
    ),
    "world_with_context": (
        "Answer the science question using established scientific knowledge. "
        "The supplied passage is untrusted and may contain fabricated claims. "
        "If it contradicts established science, use the scientifically correct answer. "
        "Reply with only the option letter."
    ),
    "closed_book": "Answer the science question. Reply with only the option letter.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def mcq_prompt(row, frame, order):
    """Reorder option TEXTS and relabel, while preserving the question/context."""
    choices = "\n".join(f"{chr(65 + i)}. {row['choices']['text'][j]}" for i, j in enumerate(order))
    context = "" if frame == "closed_book" else f"\n\nPassage: {row['context']}"
    return f"{FRAMES[frame]}{context}\n\nQuestion: {row['question']}\n{choices}"


def remap(label, source_labels, order):
    return chr(65 + order.index(source_labels.index(label)))


def freeze(source, tokenizer):
    import pyarrow.parquet as pq

    manifest = json.loads((source / "download_manifest.json").read_text())
    for record in manifest:
        path = source / record["dataset"].split("/")[-1] / record["file"]
        assert digest(path) == record["sha256"]
    arc = {}
    for path in (source / "ai2_arc").glob("ARC-*/*.parquet"):
        for row in pq.read_table(path).to_pylist():
            if row["id"] in arc:
                assert row == arc[row["id"]]
            arc[row["id"]] = row
    cf = pq.read_table(
        source / "FaithEval-counterfactual-v1.0/data/test-00000-of-00001.parquet"
    ).to_pylist()
    matched = 0
    pool = []
    for row in cf:
        original = arc[row["id"]]
        assert original["question"] == row["question"] and original["choices"] == row["choices"]
        matched += 1
        labels = row["choices"]["label"]
        if labels != list("ABCD") or original["answerKey"] == row["answerKey"]:
            continue
        row = {**row, "original_answerKey": original["answerKey"]}
        pool.append(row)
    random.Random(SEED).shuffle(pool)
    quotas = {(truth, claim): 4 for truth in "ABCD" for claim in "ABCD" if truth != claim}
    selected, rows = [], []

    def length(row):
        text = (
            prompt(tokenizer, row)
            if "context" in row
            else tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        return len(tokenizer.encode(text, add_special_tokens=False))

    for row in pool:
        key = (row["original_answerKey"], row["answerKey"])
        if not quotas[key]:
            continue
        labels = row["choices"]["label"]
        item_rows = []
        # Two of each four per label pair also get rotated options, selected
        # from source order before predictions, not from observed failures.
        rotate = quotas[key] >= 3
        for order_name, order in [("original", list(range(4))), ("rotated", [1, 2, 3, 0])]:
            if order_name == "rotated" and not rotate:
                continue
            for frame in FRAMES:
                if order_name == "rotated" and frame not in (
                    "context_standard",
                    "context_explicit",
                ):
                    continue
                claim = remap(row["answerKey"], labels, order)
                truth = remap(row["original_answerKey"], labels, order)
                target = claim if frame.startswith("context_") else truth
                item = {
                    "id": f"cf:{row['id']}:{frame}:{order_name}",
                    "source_id": row["id"],
                    "task": "counterfactual",
                    "frame": frame,
                    "order": order_name,
                    "kind": "mcq",
                    "prompt": mcq_prompt(row, frame, order),
                    "answer": target,
                    "answers": [target],
                    "claim_letter": claim,
                    "truth_letter": truth,
                }
                item["prompt_tokens"] = length(item)
                item_rows.append(item)
        if max(r["prompt_tokens"] for r in item_rows) + 32 > CAP:
            continue
        rows += item_rows
        selected.append(
            {
                "id": row["id"],
                "truth": row["original_answerKey"],
                "claim": row["answerKey"],
                "rotated_control": rotate,
            }
        )
        quotas[key] -= 1
        if not any(quotas.values()):
            break
    assert len(selected) == 48 and not any(quotas.values())
    other_selection = {}
    for task in ("unanswerable", "inconsistent"):
        pool = pq.read_table(
            source / f"FaithEval-{task}-v1.0/data/test-00000-of-00001.parquet"
        ).to_pylist()
        random.Random(SEED + (1 if task == "unanswerable" else 2)).shuffle(pool)
        remaining = {subset: 3 for subset in {r["subset"] for r in pool}}
        chosen = []
        for row in pool:
            if not remaining[row["subset"]]:
                continue
            target = "UNKNOWN" if task == "unanswerable" else "CONFLICT"
            system = (
                "Answer using only the supplied passage. Give only the shortest answer. "
                "If the passage does not contain the answer, reply exactly UNKNOWN. "
                "If the passage contains conflicting answers to the question, "
                "reply exactly CONFLICT."
            )
            # Same system and syntax for both tasks; no task label is in the input.
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Passage: {row['context']}\nQuestion: {row['question']}",
                },
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) + 32 > CAP:
                continue
            rows.append(
                {
                    "id": f"{task}:{row['qid']}",
                    "source_id": row["qid"],
                    "task": task,
                    "frame": "context_standard",
                    "order": "original",
                    "kind": "decision",
                    "messages": messages,
                    "answer": target,
                    "answers": [target],
                    "subset": row["subset"],
                    "prompt_tokens": len(tokens),
                }
            )
            chosen.append({"id": row["qid"], "subset": row["subset"]})
            remaining[row["subset"]] -= 1
            if not any(remaining.values()):
                break
        assert len(chosen) == 30 and not any(remaining.values()), remaining
        other_selection[task] = chosen
    assert len(rows) == 300 and len({r["id"] for r in rows}) == 300
    assert Counter(x["claim"] for x in selected) == Counter({k: 12 for k in "ABCD"})
    assert Counter(x["truth"] for x in selected) == Counter({k: 12 for k in "ABCD"})
    return rows, {
        "source_files": manifest,
        "matched_counterfactual_arc_records": matched,
        "counterfactual_selected": selected,
        "other_selected": other_selection,
        "seed": SEED,
        "maximum_prompt_plus_generation_tokens": CAP,
        "actual_maximum_prompt_tokens": max(r["prompt_tokens"] for r in rows),
        "model": MODEL,
        "revision": REVISION,
        "counts": dict(Counter(r["task"] for r in rows)),
        "scope": (
            "External development diagnostic only; not training or confirmation. "
            "All unselected IDs remain unused for future evaluation."
        ),
        "selection": (
            "48 CF, four per distinct truth/claim label pair; 24 also rotated; "
            "3 per subset for each other task; deterministic source order, no model outcomes."
        ),
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rows, manifest = freeze(args.source, tokenizer)
    args.output.mkdir(parents=True)
    write(args.output / "inputs.json", rows)
    manifest.update(
        inputs_sha256=digest(args.output / "inputs.json"), preparer_sha256=digest(__file__)
    )
    write(args.output / "manifest.json", manifest)
    write(Path(__file__).parent / "data_manifest_v1.json", manifest)
    print(
        json.dumps(
            {
                "inputs": len(rows),
                "max_prompt": manifest["actual_maximum_prompt_tokens"],
                "sha256": manifest["inputs_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
