import runpy
from pathlib import Path

helpers = runpy.run_path(str(Path(__file__).parents[1] / "experiments/real_qa/data.py"))
normalize = helpers["normalize"]
partition_articles = helpers["partition_articles"]
score = helpers["score"]


def test_alias_scoring_and_partial_answer():
    assert score("The United States.", ["USA", "United States"])["em"] == 1
    assert score("New York", ["New York City"])["f1"] == 0.8
    assert score("red red", ["red blue"])["f1"] == 0.5
    assert normalize("The 1940s") == "1940s"
    assert score("UNKNOWN.", [])["em"] == 1
    assert score("I think UNKNOWN", [])["em"] == 0
    assert score("UNKNOWN", ["Paris"])["em"] == 0


def test_article_partition_is_disjoint_and_order_independent():
    articles = [{"title": str(i)} for i in range(100)]
    splits = partition_articles(articles)
    assert splits == partition_articles(list(reversed(articles)))
    titles = [a["title"] for rows in splits.values() for a in rows]
    assert len(titles) == len(set(titles)) == 100
    assert [len(splits[k]) for k in ("train", "cal", "val")] == [20, 40, 40]


def test_pairs_keep_full_context_and_filter_instead_of_truncating():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[1]["content"]

        def encode(self, text, **kwargs):
            return text.split()

    def paragraph(context, prefix):
        return {
            "context": context,
            "qas": [
                {
                    "id": prefix + "yes",
                    "question": "Where?",
                    "is_impossible": False,
                    "answers": [{"text": "Paris"}, {"text": "the city of Paris"}],
                },
                {"id": prefix + "no", "question": "When?", "is_impossible": True, "answers": []},
            ],
        }

    articles = [
        {
            "title": "France",
            "paragraphs": [
                paragraph("The city is Paris.", "short"),
                paragraph("word " * 100, "long"),
            ],
        }
    ]
    rows = helpers["paired_rows"](articles, Tokenizer(), pairs=1, seed=1, max_tokens=20)
    assert {r["id"] for r in rows} == {"shortyes", "shortno"}
    assert all(r["context"] == "The city is Paris." for r in rows)
    assert rows[0]["answers"] == ["Paris", "the city of Paris"]
