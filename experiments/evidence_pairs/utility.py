"""Freeze an unrelated generative multiple-choice diagnostic before model use."""

import argparse
import hashlib
import json
import random
from pathlib import Path

REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
SOURCE_SHA = "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SOURCE_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        local_files_only=True,
    )
    pool = pq.read_table(args.source).to_pylist()
    random.Random(20261022).shuffle(pool)
    quotas = {(domain, label): 8 for domain in ("activitynet", "wikihow") for label in range(4)}
    selected, used = [], set()
    for row in pool:
        domain = row["source_id"].split("~")[0]
        key = (domain, int(row["label"]))
        if not quotas.get(key) or row["source_id"] in used:
            continue
        choices = "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(row["endings"]))
        prompt = (
            "Choose the most plausible continuation of the situation. "
            "Reply with only A, B, C, or D.\n\nSituation: " + row["ctx"] + "\n\n" + choices
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        if len(ids) + 2 > 512:
            continue
        answer = chr(65 + int(row["label"]))
        selected.append(
            {
                "id": f"hellaswag_val_{row['ind']}",
                "kind": "hellaswag",
                "prompt": prompt,
                "answer": answer,
                "answers": [answer],
                "source_id": row["source_id"],
                "source_index": row["ind"],
                "domain": domain,
                "prompt_tokens": len(ids),
            }
        )
        used.add(row["source_id"])
        quotas[key] -= 1
        if not any(quotas.values()):
            break
    assert len(selected) == 64 and not any(quotas.values())
    encoded = json.dumps(selected, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    manifest = {
        "source": "https://huggingface.co/datasets/Rowan/hellaswag",
        "revision": REVISION,
        "source_file": "data/validation-00000-of-00001.parquet",
        "source_sha256": SOURCE_SHA,
        "license": "MIT as listed in the dataset card; preserve upstream attribution",
        "sampling": (
            "seed 20261022, 8 per answer letter per domain, one per source_id, <=512 tokens"
        ),
        "scope": (
            "64-question generative MCQ diagnostic; NOT the official HellaSwag likelihood score"
        ),
        "use": "evaluation only; no training, neuron selection or hyperparameter selection",
        "prepared_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected": [
            {k: row[k] for k in ("id", "source_id", "source_index", "domain")} for row in selected
        ],
    }
    (Path(__file__).parent / "utility_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"rows": len(selected), "sha256": manifest["prepared_sha256"]}))


if __name__ == "__main__":
    main()
