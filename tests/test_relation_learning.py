import hashlib
import json

import pytest
import torch

from experiments.relation_learning.prepare import build, protocol


def test_replay_flags_changed_adapter_even_when_metrics_match(tmp_path):
    import shutil

    from safetensors.torch import save_file

    from experiments.relation_learning.replay import compare

    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    label = "lora-43-0.0003"
    record = {"case": 0, "condition": {"id": label}, "history": [{"step": 1}], "metrics": {}}
    (before / "fits.json").write_text(json.dumps([record]))
    for name in ("inputs.json", "tokenization.json", "case-0-base.json", f"case-0-{label}.json"):
        (before / name).write_text("{}")
    adapter = f"case-0-{label}-adapter/adapter_model.safetensors"
    (before / adapter).parent.mkdir()
    for name in (adapter, f"case-0-{label}-teacher.safetensors", "case-0-scores.safetensors"):
        save_file({"value": torch.tensor([1.0, 2.0])}, str(before / name))
    shutil.copytree(before, after)
    assert compare(before, after)["exact_replay"]
    save_file({"value": torch.tensor([1.0, 3.0])}, str(after / adapter))
    result = compare(before, after)
    assert not result["exact_replay"] and result["metrics_exact"]
    assert result["tensors"][adapter]["max_abs_difference"] == 1


def test_review_distinguishes_corrections_damage_and_incomplete_answers():
    from experiments.relation_learning.review import assemble

    inputs = [{"id": str(i), "prompt": "Who?", "answers": ["Mira"]} for i in range(3)]

    def output(i, text, ended=True):
        return {
            "id": str(i),
            "text": text,
            "ended": ended,
            "truncated": not ended,
            "unexpected_special": False,
        }

    base = {"target_test": [output(0, "Oren"), output(1, "Mira"), output(2, "Mira")]}
    after = {"target_test": [output(0, "Mira"), output(1, "UNKNOWN"), output(2, "Mira", False)]}
    rows, stats = assemble({"target_test": inputs}, base, after, condition="test", selected=True)
    assert stats[0]["corrections"] == 1
    assert stats[0]["regressions"] == 2
    assert stats[0]["base_exact"] == 2 and stats[0]["adapted_exact"] == 1
    assert stats[0]["adapted_truncated"] == 1
    assert all(row["semantic_review"] is None for row in rows)
    after["target_test"].reverse()
    with pytest.raises(ValueError):
        assemble({"target_test": inputs}, base, after, condition="test", selected=True)


def test_validation_selection_ignores_test_results_and_breaks_ties():
    from experiments.relation_learning.select import choose

    rows = [
        {
            "condition": {"family": "lora", "seed": 43, "lr": lr, "id": str(lr)},
            "target_val_mean_nll": 1.0,
            "test_accuracy": score,
        }
        for lr, score in ((0.001, 1.0), (0.0003, 0.0))
    ]
    expected = choose(rows)
    assert expected[0]["lr"] == 0.0003
    for row in rows:
        row["test_accuracy"] = 1 - row["test_accuracy"]
    assert choose(rows) == expected
    rows[0]["target_val_mean_nll"] = 0.9
    assert choose(rows)[0]["lr"] == 0.001
    with pytest.raises(ValueError):
        choose(rows + rows[:1])


def test_learning_splits_and_discovery_do_not_use_evaluation():
    data = build()["cases"][0]
    spec = protocol("unused")
    assert len(data["train"]) == 64 and len(data["preserve"]) == 20
    assert len(data["target_val"]) == 32 and len(data["target_test"]) == 64
    assert {r["world"] for r in data["train"]}.isdisjoint(
        {r["world"] for r in data["target_val"] + data["target_test"]}
    )
    assert {r["world"] for r in data["target_val"]}.isdisjoint(
        {r["world"] for r in data["target_test"]}
    )
    discovery = [data["train"][i] for i in spec["discovery_indices"]]
    assert {r["world"] for r in discovery} == {0, 1, 2, 3}
    assert sum(r["condition"] == "missing" for r in discovery) == 4
    assert {r["family"] for r in discovery} == {"interview", "release"}
    assert all(r["variant"] == "original" for r in data["train"])
    assert sum(r["variant"] == "transfer" for r in data["target_test"]) == 32
    ids = [r["id"] for bank, rows in data.items() if isinstance(rows, list) for r in rows]
    assert len(ids) == len(set(ids))


def test_empirical_score_does_not_square_the_average_gradient():
    from experiments.paired_mlp.selection import add_empirical_group_scores, projection_scores

    a, b = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0, 5.0]])
    c, d = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[4.0, 5.0]])
    capture = {"gate": (a, b), "up": (a, b), "down": (c, d)}
    opposite = {k: (x, -y) for k, (x, y) in capture.items()}
    groups = [{"incoming": ("gate", "up"), "outgoing": "down"}]
    scores = add_empirical_group_scores(None, capture, groups)
    scores = add_empirical_group_scores(scores, opposite, groups)
    # Independent dense gradients: row norms for gate/up, column norms for down.
    expected = 2 * (b.T @ a).square().sum(1) + (d.T @ c).square().sum(0)
    torch.testing.assert_close(scores[0]["paired_empirical_fisher"] / 2, expected.double())
    mean = projection_scores([capture, opposite], incoming=("gate", "up"), outgoing="down")
    torch.testing.assert_close(mean["paired_gradient_norm"], torch.zeros(3, dtype=torch.float64))


def test_actual_worker_fisher_random_lora_with_custom_messages(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from experiments.paired_mlp.run import execute
    from experiments.paired_mlp.verify import verify
    from neuronpatch.patch import fingerprint
    from tests.test_paired_worker import TinyTokenizer

    data = build()
    # Small bank subsets retain the full training/discovery index layout.
    for bank in (
        "preserve",
        "target_val",
        "target_test",
        "scope_val",
        "scope_test",
        "scope_exposed",
        "utility_val",
        "utility_test",
    ):
        data["cases"][0][bank] = data["cases"][0][bank][:2]
    raw = (json.dumps(data, indent=2) + "\n").encode()
    spec = protocol(hashlib.sha256(raw).hexdigest())
    spec.update(
        architecture={"layers": 2, "hidden": 16, "intermediate": 32},
        steps=1,
        max_new_tokens=2,
        discovery_indices=[1, 10, 41, 50],
        minimum_loss_scale=1,
    )
    spec["conditions"] = [spec["conditions"][i] for i in (0, 2, 6)]
    spec["new_fits"] = 3
    config = transformers.Qwen3Config(
        vocab_size=1024,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(43)
    model = transformers.Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    before = fingerprint(model)
    tokenizer = TinyTokenizer()
    fitted = execute(model, tokenizer, data, spec, tmp_path / "run", lambda: None)
    assert len(fitted) == 3 and fingerprint(model) == before
    assert all(x["base_restored"] and x["reload_teacher_exact"] for x in fitted)
    checked = verify(tmp_path / "run", spec, tokenizer)
    assert checked["fits"] == 3
    discovery = json.loads((tmp_path / "run/case-0-discovery.json").read_text())
    assert discovery["backward_passes"] == 4
    assert discovery["training_indices"] == [1, 10, 41, 50]
    # A result with different training examples must fail independent verification.
    path = tmp_path / "run/fits.json"
    original = path.read_text()
    altered = json.loads(original)
    altered[0]["history"][0]["example_indices"][0] = -1
    path.write_text(json.dumps(altered))
    with pytest.raises(AssertionError):
        verify(tmp_path / "run", spec, tokenizer)
    path.write_text(original)
    path = tmp_path / "run/case-0-discovery.json"
    original = path.read_text()
    altered = json.loads(original)
    altered["training_indices"] = [0, 1, 2, 3]
    path.write_text(json.dumps(altered))
    with pytest.raises(AssertionError):
        verify(tmp_path / "run", spec, tokenizer)
    path.write_text(original)
