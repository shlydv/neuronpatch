"""Article-isolated SQuAD pairs and alias-aware evaluation (experiment helpers)."""

import collections
import random
import re
import string

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = (
    "Answer using only the supplied passage. Give only the shortest answer span. "
    "If the passage does not contain the answer, reply exactly UNKNOWN."
)


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(prediction, answers):
    """SQuAD-style EM/F1 over all aliases; explicit UNKNOWN for impossible items."""
    abstained = prediction.strip().rstrip(".!?").upper() == "UNKNOWN"
    if not answers:
        return {"em": float(abstained), "f1": float(abstained), "abstained": abstained}
    pred = normalize(prediction).split()
    scores = []
    for answer in answers:
        gold = normalize(answer).split()
        overlap = sum((collections.Counter(pred) & collections.Counter(gold)).values())
        f1 = 2 * overlap / (len(pred) + len(gold)) if pred and gold else float(pred == gold)
        scores.append((float(pred == gold), f1))
    return {
        "em": max(s[0] for s in scores),
        "f1": max(s[1] for s in scores),
        "abstained": abstained,
    }


def prompt(tokenizer, row):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Passage: {row['context']}\nQuestion: {row['question']}",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def partition_articles(articles, seed=20261019):
    ordered = sorted(articles, key=lambda x: x["title"])
    random.Random(seed).shuffle(ordered)
    # Entire articles, including every paragraph/question, belong to a single split.
    return {"train": ordered[80:], "cal": ordered[:40], "val": ordered[40:80]}


def paired_rows(articles, tokenizer, pairs, seed, max_tokens=512):
    """Sample one supported/impossible pair per paragraph, without truncation."""
    rng = random.Random(seed)
    candidates = []
    for article in sorted(articles, key=lambda x: x["title"]):
        for paragraph in article["paragraphs"]:
            groups = {False: [], True: []}
            for qa in paragraph["qas"]:
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                if bool(answers) == qa["is_impossible"]:
                    raise ValueError("Answerability does not match gold answers")
                row = {
                    "id": qa["id"],
                    "article": article["title"],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": answers,
                    "answer": answers[0] if answers else "UNKNOWN",
                    "kind": "answerable" if answers else "missing",
                }
                length = len(tokenizer.encode(prompt(tokenizer, row), add_special_tokens=False))
                target_length = len(tokenizer.encode(row["answer"], add_special_tokens=False))
                if length + target_length + 1 <= max_tokens and target_length <= 24:
                    row["prompt_tokens"] = length
                    groups[qa["is_impossible"]].append(row)
            if all(groups.values()):
                candidates.append([rng.choice(groups[False]), rng.choice(groups[True])])
    if len(candidates) < pairs:
        raise ValueError(f"Need {pairs} eligible paragraphs, found {len(candidates)}")
    rng.shuffle(candidates)
    return [row for pair in candidates[:pairs] for row in pair]


def metrics(rows):
    result = {}
    for kind in ("answerable", "missing"):
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        result[kind] = {
            "n": len(subset),
            **{key: sum(r[key] for r in subset) / len(subset) for key in ("em", "f1", "abstained")},
        }
    return result
