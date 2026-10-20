"""Freeze a new article-isolated development pilot without model predictions."""

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SOURCE_HASH = "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_snapshot = Path(__file__).read_bytes()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError("Expected the pinned official SQuAD 2.0 training source")
    spec = importlib.util.spec_from_file_location("grounded_data", ROOT.parent / "real_qa/data.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    history_paths = [
        ROOT.parent / "real_qa/scaled" / f"{name}_expanded.json" for name in ("train", "cal", "val")
    ]
    historical = [row for path in history_paths for row in json.loads(path.read_text())]
    used = {row["article"] for row in historical}
    used_contexts = {row["context"] for row in historical}
    articles = [
        {
            **article,
            "paragraphs": [
                paragraph
                for paragraph in article["paragraphs"]
                if paragraph["context"] not in used_contexts
            ],
        }
        for article in json.loads(raw)["data"]
    ]
    articles.sort(key=lambda article: article["title"])
    rng = random.Random(20261020)
    rng.shuffle(articles)
    partitions = {"train": articles[160:], "cal": articles[:80], "val": articles[80:160]}
    model = json.loads((ROOT / "model_candidate.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model["model"], revision=model["revision"])
    counts = {"train": 64, "cal": 16, "val": 64}
    rows = {
        name: helper.paired_rows(
            partitions[name], tokenizer, pairs, seed=20261020 + offset, max_tokens=512
        )
        for offset, (name, pairs) in enumerate(counts.items())
    }
    titles = {name: {row["article"] for row in group} for name, group in rows.items()}
    assert not any(titles[a] & titles[b] for a in titles for b in titles if a != b)
    assert not {row["context"] for group in rows.values() for row in group} & used_contexts
    utility = []
    for a, b in ((17, 28), (42, 19), (63, 18), (81, 14)):
        utility.extend(
            [
                {
                    "id": f"sum-{a}-{b}",
                    "prompt": f"What is {a} + {b}? Reply with only the integer.",
                    "answers": [str(a + b)],
                    "kind": "arithmetic",
                },
                {
                    "id": f"diff-{a}-{b}",
                    "prompt": f"What is {a} - {b}? Reply with only the integer.",
                    "answers": [str(a - b)],
                    "kind": "arithmetic",
                },
                {
                    "id": f"product-{a}-{b}",
                    "prompt": f"What is {a} * {b}? Reply with only the integer.",
                    "answers": [str(a * b)],
                    "kind": "arithmetic",
                },
            ]
        )
    for word in ("orchid", "raven", "meadow", "silver", "planet", "velvet", "cobalt", "willow"):
        utility.append(
            {
                "id": f"upper-{word}",
                "prompt": f"Convert {word} to uppercase. Reply with only the result.",
                "answers": [word.upper()],
                "kind": "text_operation",
            }
        )
    for word, answer in (
        ("hot", "cold"),
        ("early", "late"),
        ("empty", "full"),
        ("wide", "narrow"),
        ("ancient", "modern"),
        ("shallow", "deep"),
        ("noisy", "quiet"),
        ("rough", "smooth"),
    ):
        utility.append(
            {
                "id": f"opposite-{word}",
                "prompt": f"Give the opposite of {word}. Reply with one word.",
                "answers": [answer]
                + {
                    "empty": ["filled"],
                    "ancient": ["new", "contemporary", "recent"],
                    "noisy": ["silent"],
                    "rough": ["polished"],
                }.get(word, []),
                "kind": "antonym",
            }
        )
    for country, capital in (
        ("Japan", "Tokyo"),
        ("Italy", "Rome"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
    ):
        utility.append(
            {
                "id": f"capital-{country}",
                "prompt": f"What is the capital of {country}? Reply only with the city.",
                "answers": [capital],
                "kind": "factual",
            }
        )
    assert len(utility) == 32
    rows["utility"] = utility
    args.output.mkdir(parents=True)
    for name, data in rows.items():
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "source": "Official SQuAD 2.0 train-v2.0.json",
        "source_sha256": SOURCE_HASH,
        "license": "SQuAD-derived rows: CC BY-SA 4.0; utility prompts authored for this pilot",
        "model": model["model"],
        "revision": model["revision"],
        "selection_seed": 20261020,
        "historically_used_articles": sorted(used),
        "history_policy": (
            "All current inputs use previously unused paragraph contexts. Training articles "
            "and held-out articles may have historical 0.5B exposure; current training, "
            "calibration and validation articles are mutually disjoint."
        ),
        "historical_files_sha256": {
            str(path.relative_to(ROOT.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in history_paths
        },
        "article_partitions": {
            name: sorted(article["title"] for article in group)
            for name, group in partitions.items()
        },
        "sampled_articles": {name: sorted(values) for name, values in titles.items()},
        "counts": {name: len(data) for name, data in rows.items()},
        "sample_ids": {name: [row["id"] for row in data] for name, data in rows.items()},
        "sha256": {
            name: hashlib.sha256((args.output / f"{name}.json").read_bytes()).hexdigest()
            for name in rows
        },
        "confirmation": "official development source not read or used",
        "scope": "new-passage article-isolated development pilot, not final confirmation",
        "code_sha256": hashlib.sha256(source_snapshot).hexdigest(),
        "data_helper_sha256": hashlib.sha256(
            (ROOT.parent / "real_qa/data.py").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "sampled_articles": {name: len(group) for name, group in titles.items()},
                "available_training_articles": len(partitions["train"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
