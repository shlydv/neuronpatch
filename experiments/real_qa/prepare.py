"""Create deterministic, article-disjoint samples; never inspect model predictions."""

import hashlib
import json
from pathlib import Path

from data import MODEL, REVISION, paired_rows, partition_articles
from transformers import AutoTokenizer

ROOT = Path(__file__).parent


def main():
    source = Path("/private/tmp/neuronpatch-squad")
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    print("Source provenance:", manifest, flush=True)
    files = {name: source / f"{name}-v2.0.json" for name in ("train", "dev")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    expected = {
        "train": "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
        "dev": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    }
    if hashes != expected:
        raise ValueError("Source content changed")
    sources = {name: json.loads(path.read_text())["data"] for name, path in files.items()}
    splits = partition_articles(sources["train"])
    splits["confirmation"] = sources["dev"]
    titles = {k: {a["title"] for a in v} for k, v in splits.items()}
    for a in titles:
        for b in titles:
            if a != b and titles[a] & titles[b]:
                raise ValueError("Article leakage")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    sizes = {"train": 64, "cal": 16, "val": 16, "confirmation": 32}
    data = {
        name: paired_rows(articles, tokenizer, sizes[name], 100 + i)
        for i, (name, articles) in enumerate(splits.items())
    }
    out = ROOT / "prepared"
    out.mkdir(exist_ok=False)
    for name, rows in data.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes,
                "model": MODEL,
                "revision": REVISION,
                "license": "SQuAD 2.0: CC BY-SA 4.0; derived samples retain source license",
                "max_total_tokens": 512,
                "article_partitions": {k: sorted(v) for k, v in titles.items()},
                "split_sha256": {
                    k: hashlib.sha256((out / f"{k}.json").read_bytes()).hexdigest() for k in data
                },
            },
            indent=2,
        )
    )
    print({k: len(v) for k, v in data.items()}, flush=True)


if __name__ == "__main__":
    main()
