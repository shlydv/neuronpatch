"""Prepare auditable contrasting-evidence candidates, without model predictions.

Both variants delete one sentence-like unit. Supported deletes a length-matched
unrelated unit; missing deletes the answer-bearing unit. Lexical absence is NOT
accepted as proof of unanswerability: candidates require review before training.
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

SOURCE_SHA256 = "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sentence_spans(text):
    """Return explicit removal units, not a claim of perfect sentence parsing."""
    boundaries = [0]
    for match in re.finditer(r"[.!?][\"')\]]*(?=\s+[A-Z(\"])", text):
        prefix = text[: match.end()]
        if re.search(
            r"\b(?:[A-Za-z]\.){2,}$|\b[A-Z]\.$|\b(?:Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|vs)\.$",
            prefix,
        ):
            continue
        boundaries.append(match.end())
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return [(a, b) for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]


def answer_present(context, answer):
    context, answer = (
        unicodedata.normalize("NFKC", value).casefold() for value in (context, answer)
    )
    return re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", context) is not None


def without_unit(text, spans, removed):
    return " ".join(text[a:b].strip() for index, (a, b) in enumerate(spans) if index != removed)


def paragraph_candidate(paragraph, article, tokenizer, rng, max_tokens):
    # Import only the established, unchanged prompt, not earlier row selection.
    from experiments.real_qa.data import prompt

    text = paragraph["context"]
    if not 60 <= len(text.split()) <= 300:
        return None
    spans = sentence_spans(text)
    if len(spans) < 4 or any(len(text[a:b].split()) < 4 for a, b in spans):
        return None
    if any(text[a:b].count('"') % 2 for a, b in spans):
        return None
    unit_lengths = [
        len(tokenizer.encode(text[a:b].strip(), add_special_tokens=False)) for a, b in spans
    ]
    questions = list(paragraph["qas"])
    rng.shuffle(questions)
    for qa in questions:
        if qa["is_impossible"] or not qa["answers"]:
            continue
        # Controlled atomic-fact diagnostic, not a substitute for broad QA tests.
        # Open "why/what" questions often admit another valid answer after deletion.
        if not re.match(r"^(Who|When|Where)\b", qa["question"]):
            continue
        if not re.search(r"\b[A-Z][A-Za-z]+\b|\d", qa["question"].split(" ", 1)[-1]):
            continue
        answers = list(dict.fromkeys(answer["text"] for answer in qa["answers"]))
        if any(
            not 4 <= len(answer) <= 64 or answer_present(qa["question"], answer)
            for answer in answers
        ):
            continue
        covered = {
            index
            for answer in qa["answers"]
            for index, (a, b) in enumerate(spans)
            if a <= answer["answer_start"] and answer["answer_start"] + len(answer["text"]) <= b
        }
        if len(covered) != 1:
            continue
        target = next(iter(covered))
        a, b = spans[target]
        if any(
            not a <= answer["answer_start"] or answer["answer_start"] + len(answer["text"]) > b
            for answer in qa["answers"]
        ):
            continue
        missing = without_unit(text, spans, target)
        if any(answer_present(missing, answer) for answer in answers):
            continue
        # Reject surviving informative name/date fragments as well as the full span.
        fragments = {
            word
            for answer in answers
            for word in re.findall(r"\b[A-Z][A-Za-z]+\b|\b\d+\b", answer)
            if word not in {"The", "A", "An", "United", "States", "New", "North", "South"}
        }
        if not fragments or any(answer_present(missing, word) for word in fragments):
            continue
        controls = [index for index in range(len(spans)) if index != target]
        rng.shuffle(controls)
        control = min(controls, key=lambda index: abs(unit_lengths[index] - unit_lengths[target]))
        supported = without_unit(text, spans, control)
        if any(
            context.count(left) != context.count(right)
            for context in (supported, missing)
            for left, right in (("(", ")"), ("[", "]"))
        ):
            continue
        variants = {}
        for kind, context in (("answerable", supported), ("missing", missing)):
            row = {
                "id": f"{qa['id']}:{kind}",
                "pair_id": qa["id"],
                "article": article,
                "question": qa["question"],
                "context": context,
                "answers": answers if kind == "answerable" else [],
                "answer": answers[0] if kind == "answerable" else "UNKNOWN",
                "kind": kind,
            }
            row["prompt_tokens"] = len(
                tokenizer.encode(prompt(tokenizer, row), add_special_tokens=False)
            )
            target_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
            if len(target_ids) > 16 or row["prompt_tokens"] + len(target_ids) + 1 > max_tokens:
                break
            variants[kind] = row
        if len(variants) != 2:
            continue
        length_delta = (
            variants["answerable"]["prompt_tokens"] - variants["missing"]["prompt_tokens"]
        )
        if abs(length_delta) > 8:
            continue
        return {
            "pair_id": qa["id"],
            "article": article,
            "question": qa["question"],
            "answers": answers,
            "original_context_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "original_context": text,
            "removal_spans": spans,
            "answer_unit": target,
            "control_unit": control,
            "answer_sentence": text[slice(*spans[target])].strip(),
            "control_sentence": text[slice(*spans[control])].strip(),
            "prompt_length_delta": length_delta,
            "answer_first_token": tokenizer.encode(answers[0], add_special_tokens=False)[0],
            "variants": variants,
            "review_status": "UNREVIEWED; lexical absence alone is not semantic evidence",
        }
    return None


def historical_contexts():
    paths = sorted((ROOT / "experiments/real_qa/prepared").glob("*.json"))
    paths += sorted((ROOT / "experiments/real_qa/scaled").glob("*_expanded.json"))
    for name in ("data", "data_narrow_article_attempt"):
        paths += sorted((ROOT / "artifacts/protected_rows_pilot1" / name).glob("*.json"))
    contexts = set()
    for path in paths:
        rows = json.loads(path.read_text())
        if isinstance(rows, list):
            contexts.update(row["context"] for row in rows if "context" in row)
    return contexts, {str(path.relative_to(ROOT)): digest(path) for path in paths}


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.source) != SOURCE_SHA256:
        raise ValueError("Pinned official SQuAD2 training source required")
    if args.candidates < 1:
        raise ValueError("Positive candidate count required")
    history, hashes = historical_contexts()
    articles = sorted(json.loads(args.source.read_text())["data"], key=lambda row: row["title"])
    random.Random(20261021).shuffle(articles)
    partitions = {"cal": articles[:90], "val": articles[90:180], "train": articles[180:]}
    model = json.loads((ROOT / "experiments/protected_rows/model_candidate.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        model["model"], revision=model["revision"], local_files_only=True
    )
    values = {}
    for offset, (name, group) in enumerate(partitions.items()):
        rng = random.Random(20261021 + offset)
        queues = []
        for article in group:
            paragraphs = [p for p in article["paragraphs"] if p["context"] not in history]
            rng.shuffle(paragraphs)
            queues.append((article["title"], paragraphs))
        candidates = []
        # Round-robin articles: large articles cannot consume the candidate pool.
        while len(candidates) < args.candidates and any(queue for _, queue in queues):
            for title, queue in queues:
                while queue:
                    result = paragraph_candidate(queue.pop(), title, tokenizer, rng, 512)
                    if result is not None:
                        candidates.append(result)
                        break
                if len(candidates) == args.candidates:
                    break
        if len(candidates) < args.candidates:
            raise ValueError(f"{name}: only {len(candidates)} eligible new contexts")
        values[name] = candidates
    assert len({row["pair_id"] for rows in values.values() for row in rows}) == sum(
        map(len, values.values())
    )
    titles = {name: {row["article"] for row in rows} for name, rows in values.items()}
    assert not any(titles[a] & titles[b] for a in titles for b in titles if a != b)
    args.output.mkdir(parents=True)
    for name, rows in values.items():
        write(args.output / f"{name}_candidates.json", rows)
    write(
        args.output / "manifest.json",
        {
            "status": "candidates only; semantic review required; no model predictions",
            "source": "Official SQuAD 2.0 train-v2.0.json; CC BY-SA 4.0",
            "source_sha256": SOURCE_SHA256,
            "model": model["model"],
            "revision": model["revision"],
            "prompt_helper_sha256": digest(ROOT / "experiments/real_qa/data.py"),
            "prepare_sha256": digest(Path(__file__)),
            "historical_files_sha256": hashes,
            "historical_contexts_excluded": len(history),
            "selection_seed": 20261021,
            "maximum_prompt_plus_answer_tokens": 512,
            "maximum_pair_prompt_length_difference": 8,
            "scope": (
                "Controlled atomic-fact development diagnostic (who/when/where); "
                "not representative broad QA or a replacement for fresh transfer evaluation"
            ),
            "counts": {name: len(rows) for name, rows in values.items()},
            "sampled_articles": {name: sorted(group) for name, group in titles.items()},
            "sha256": {name: digest(args.output / f"{name}_candidates.json") for name in values},
            "confirmation": "Official SQuAD development and FaithEval have not been read or used",
            "history_limit": "New paragraphs; topics may have historical development exposure",
        },
    )
    print(
        json.dumps(
            {
                name: {"pairs": len(rows), "articles": len(titles[name])}
                for name, rows in values.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
