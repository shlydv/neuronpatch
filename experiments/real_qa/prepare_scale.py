"""Prepare nested training sizes and expanded development sets, without predictions."""

import hashlib
import json
from pathlib import Path

from data import MODEL, REVISION, paired_rows, partition_articles
from transformers import AutoTokenizer

ROOT = Path(__file__).parent


def main():
    path = Path("/private/tmp/neuronpatch-squad/train-v2.0.json")
    expected = json.loads((ROOT / "prepared/manifest.json").read_text())["source_sha256"]["train"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Training source changed")
    partitions = partition_articles(json.loads(path.read_text())["data"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    outputs = {}
    for i, name in enumerate(("train", "cal", "val")):
        rows = paired_rows(partitions[name], tokenizer, 1024 if name == "train" else 128, 100 + i)
        original = json.loads((ROOT / "prepared" / f"{name}.json").read_text())
        if rows[: len(original)] != original:
            raise ValueError("Expanded sample does not preserve original prefix")
        outputs[name + "_expanded"] = rows
        outputs[name + "_new"] = rows[len(original) :]
        if name == "train":
            outputs["train_512"] = rows[:512]
    out = ROOT / "scaled"
    out.mkdir(exist_ok=False)
    for name, rows in outputs.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": actual,
                "model": MODEL,
                "revision": REVISION,
                "license": "CC BY-SA 4.0; see ../DATA_LICENSE.md",
                "max_total_tokens": 512,
                "confirmation": "unchanged; not accessed by this script",
                "counts": {k: len(v) for k, v in outputs.items()},
                "sha256": {
                    k: hashlib.sha256((out / f"{k}.json").read_bytes()).hexdigest() for k in outputs
                },
                "status": "data prepared only; no model predictions",
            },
            indent=2,
        )
    )
    print({k: len(v) for k, v in outputs.items()}, flush=True)


if __name__ == "__main__":
    main()
