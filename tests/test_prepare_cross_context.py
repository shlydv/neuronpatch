import json

import pytest

from experiments.instruction_boundary.prepare_cross_context import prepare, sha256
from experiments.instruction_boundary.teacher_review import review_identity


def inputs(tmp_path):
    teacher = {"id": "a:clean:none", "tokens": [1, 2], "text": "answer"}
    record = {"task": {"document_id": "a"}, "teacher": teacher}
    files = {
        "teachers-train.json": [teacher],
        "inventory.json": {"train": [record]},
        "review.json": [
            {
                "split": "train",
                "document_id": "a",
                "identity": review_identity(record),
                "verdict": "accept",
                "reviewer": "fixture",
                "rationale": "fixture only",
            }
        ],
        "protocol.json": {"model": "model", "revision": "revision"},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    parent = {
        **files["protocol.json"],
        "reused_parents": [
            {
                "files_sha256": {"teachers-train.json": sha256(tmp_path / "teachers-train.json")},
                "kernel": "recorded-origin",
                "kernel_version": 1,
                "base_fingerprint": "base",
            }
        ],
    }
    (tmp_path / "parent.json").write_text(json.dumps(parent))
    return (
        tmp_path / "inventory.json",
        tmp_path / "review.json",
        tmp_path / "parent.json",
        tmp_path,
        tmp_path / "protocol.json",
    )


def test_preparation_retains_review_and_original_trajectory(tmp_path):
    result = prepare(*inputs(tmp_path))
    assert result["recorded_base_fingerprint"] == "base"
    assert result["teachers"]["train"][0]["teacher"]["tokens"] == [1, 2]
    assert len(result["sources"]) == 5
    assert result["review_receipt"][0]["verdict"] == "accept"


@pytest.mark.parametrize("mutation", ["source", "inventory", "revision", "ambiguous_origin"])
def test_preparation_rejects_broken_provenance(tmp_path, mutation):
    args = inputs(tmp_path)
    filename = {
        "source": "teachers-train.json",
        "inventory": "inventory.json",
        "revision": "protocol.json",
        "ambiguous_origin": "parent.json",
    }[mutation]
    path = tmp_path / filename
    data = json.loads(path.read_text())
    if mutation == "source":
        data[0]["text"] = "changed"
    elif mutation == "inventory":
        data["train"][0]["teacher"]["text"] = "changed"
    elif mutation == "revision":
        data["revision"] = "different"
    else:
        data["reused_parents"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepare(*args)
