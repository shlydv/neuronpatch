"""Common-objective gradients, calibration selection and actual worker lifecycle."""

from copy import deepcopy

import pytest
import torch

from experiments.adapter_comparison.analysis import choose_rates
from experiments.adapter_comparison.fit import capture_references, correction, fit_shared
from experiments.fine_transfer.lora_control import attach_gptj_lora
from experiments.subject_binding.fit import answer_kl, fit_binding
from neuronpatch.editing import GPTJEditConfig
from neuronpatch.occurrence import OccurrenceColumnEdit


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        tokens = [7, 8] if text == " new answer" else [3, 4, 5]
        if "elsewhere" in text:
            tokens = [3, 4, 6, 2]
        if "other" in text:
            tokens = [3, 9, 6, 2]
        return torch.tensor([tokens]) if kwargs.get("return_tensors") == "pt" else tokens

    def decode(self, tokens, skip_special_tokens=True):
        return str([t for t in tokens if not skip_special_tokens or t != self.eos_token_id])


def tiny():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    torch.manual_seed(721)
    return (
        transformers.GPTJForCausalLM(
            transformers.GPTJConfig(
                vocab_size=19,
                n_positions=32,
                n_embd=16,
                n_layer=4,
                n_head=2,
                n_inner=24,
                rotary_dim=8,
            )
        )
        .eval()
        .requires_grad_(False)
    )


TARGETS = [("transformer.h.0.mlp.fc_out", i) for i in (2, 7, 2, 3)]


def test_missing_adapter_dependency_fails_before_model_download(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import json
    import sys

    from experiments.adapter_comparison import run

    data = tmp_path / "data"
    data.mkdir()
    (data / "protocol.json").write_text(
        json.dumps({"runtime": {"torch": torch.__version__.split("+")[0]}})
    )

    def import_adapter(name):
        if name == "peft":
            raise ModuleNotFoundError("simulated missing setuptools in adapter import")
        return object()

    def download(*args, **kwargs):
        pytest.fail("Dependency failure must precede model/tokenizer downloads")

    monkeypatch.setattr(run.importlib, "import_module", import_adapter)
    monkeypatch.setattr(run.GPTJForCausalLM, "from_pretrained", download)
    monkeypatch.setattr(run.GPT2Tokenizer, "from_pretrained", download)
    monkeypatch.setattr(
        sys, "argv", ["run", "--data", str(data), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ModuleNotFoundError, match="setuptools"):
        run.main()


def test_shared_sparse_training_matches_existing_zero_norm_objective():
    first = tiny()
    second = deepcopy(first)
    config = GPTJEditConfig(
        max_steps=3,
        gamma=0,
        early_stop_probability=None,
        delta_dtype=torch.float32,
    )
    expected, old_history, _ = fit_binding(
        first,
        Tokenizer(),
        ["Subject is", "Subject elsewhere"],
        "new answer",
        TARGETS,
        ["other one", "Different subject"],
        10,
        config,
    )
    refs = capture_references(
        second,
        Tokenizer(),
        ["Subject is", "Subject elsewhere"],
        "new answer",
        ["other one", "Different subject"],
    )
    actual = OccurrenceColumnEdit(second, TARGETS, delta_dtype=torch.float32).eval()
    history = fit_shared(
        actual,
        actual.trainable_parameters(),
        refs,
        steps=3,
        lr=config.lr,
        weight=10,
        alpha=config.alpha,
        beta=config.beta,
        check_time=lambda: None,
    )
    torch.testing.assert_close(actual.delta, expected.delta, atol=0, rtol=0)
    assert [r["nll"] for r in history] == [r["nll"] for r in old_history]


def test_lora_sequential_preservation_gradients_match_joint_objective():
    first = tiny()
    second = deepcopy(first)
    originals = [(p, p.detach().clone()) for p in first.parameters()]
    refs = capture_references(
        first,
        Tokenizer(),
        ["Subject is"],
        "new answer",
        ["other one", "Different subject"],
    )
    actual, _ = attach_gptj_lora(first)
    expected, _ = attach_gptj_lora(second)
    fit_shared(
        actual,
        [p for p in actual.parameters() if p.requires_grad],
        refs,
        steps=3,
        lr=0.001,
        weight=10,
        alpha=1,
        beta=10,
        check_time=lambda: None,
    )
    optimizer = torch.optim.Adam([p for p in expected.parameters() if p.requires_grad], lr=0.001)
    for _ in range(3):
        task, _ = correction(expected, refs, alpha=1, beta=10)
        preservation = [
            answer_kl(expected(input_ids=ids, use_cache=False).logits[0, start:], reference)
            for ids, start, reference in refs.protected
        ]
        (task + 10 * torch.stack(preservation).mean()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for (n, p), (m, q) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert n == m
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-5)
    assert all(torch.equal(p, before) and p.grad is None for p, before in originals)


def test_rate_selection_requires_both_cases_and_does_not_receive_evaluation():
    conditions = [
        {"id": "low", "family": "lora", "lr": 0.0003},
        {"id": "high", "family": "lora", "lr": 0.001},
    ]
    fits = [
        {"case": c, "condition": name, "final": {"nll": 0.05, "max_answer_kl": 0.05}}
        for c in (0, 1)
        for name in ("low", "high")
    ]
    threshold = {"target_probability": 0.9, "max_answer_kl": 0.1}
    assert choose_rates(fits, conditions, [0, 1], threshold) == {"lora": "low"}
    fits[0]["final"]["max_answer_kl"] = 0.2
    assert choose_rates(fits, conditions, [0, 1], threshold) == {"lora": "high"}
    fits[1]["final"]["nll"] = float("nan")
    assert choose_rates(fits, conditions, [0, 1], threshold) == {"lora": None}
    with pytest.raises(ValueError, match="per case"):
        choose_rates(fits[:-1], conditions, [0, 1], threshold)


def test_complete_worker_restores_base_and_round_trips_adapters(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import hashlib
    import json
    import sys

    from experiments.adapter_comparison import run
    from experiments.adapter_comparison.analysis import summarize
    from experiments.fine_transfer.run import merge, restore, save_columns
    from experiments.fine_transfer.scope_analysis import digest_output
    from experiments.subject_binding.fit import prompt_ids
    from neuronpatch.patch import fingerprint

    model, tokenizer = tiny().half(), Tokenizer()
    identity = fingerprint(model)
    data = tmp_path / "data"
    data.mkdir()

    @torch.no_grad()
    def generate(base, tok, prompt):
        ids = prompt_ids(tok, prompt, base.device)
        output = base.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        tokens = output[0, ids.shape[1] :].tolist()
        return {"text": tok.decode(tokens), "tokens": tokens, "truncated": False}

    cases, cold = [], []
    for c in (0, 1):
        edit = OccurrenceColumnEdit(model, TARGETS, delta_dtype=torch.bfloat16)
        with torch.no_grad():
            edit.delta.normal_(std=0.01)
        edit.save(data / f"case-{c}-reference")
        probes = [{"prompt": f"reference{c}-{i}"} for i in (0, 1)]
        for p in probes:
            p["base"] = digest_output(generate(model, tokenizer, p["prompt"]))
        saved = save_columns(model, TARGETS)
        try:
            merge(model, TARGETS, edit.delta)
            for p in probes:
                p["edited"] = digest_output(generate(model, tokenizer, p["prompt"]))
        finally:
            restore(model, saved)
        cold.append({"case": c, "probes": probes})
        cases.append(
            {
                "case": c,
                "targets": [list(t) for t in TARGETS],
                "target": "new answer",
                "train_prompts": ["Subject is"],
                "edit_probes": [
                    {"kind": kind, "prompt": f"subject{c}-{i}"}
                    for i, kind in enumerate(
                        [
                            "rewrite",
                            "train/1",
                            "train/2",
                            "rephrase",
                            "fresh_wording",
                            "fresh_wording",
                        ]
                    )
                ],
                **{
                    name: [
                        {
                            "source_index": i,
                            "prompt": f"other {c}-{i}",
                            "reference_answers": "unknown",
                        }
                        for i in range(count)
                    ]
                    for name, count in (("calibration", 3), ("evaluation", 4), ("exposed_scope", 8))
                },
            }
        )
    inputs = {
        "cases": cases,
        "utility": [{"prompt": f"utility{i}", "gold": "unknown"} for i in range(32)],
    }
    for name, value in (("inputs.json", inputs), ("cold.json", cold)):
        (data / name).write_text(json.dumps(value))
    conditions = [
        {
            "id": "consensus",
            "family": "consensus",
            "lr": 0.001,
            "expected_scalars": {"0": 64, "1": 64},
        },
        {"id": "mean", "family": "mean", "lr": 0.001, "expected_scalars": {"0": 64, "1": 64}},
        *[
            {"id": name, "family": "lora", "lr": lr, "expected_scalars": {"0": 1664, "1": 1664}}
            for name, lr in (("low", 0.0003), ("high", 0.001))
        ],
    ]
    protocol = {
        "model": "tiny",
        "revision": "test",
        "base_fingerprint": identity,
        "runtime": {"torch": torch.__version__.split("+")[0]},
        "conditions": conditions,
        "lora_rank": 2,
        "seed": 43,
        "model_dimensions": {"layers": 4, "width": 16, "mlp_width": 24},
        "selection_layers_excluded": 3,
        "neurons_per_token": 2,
        "fit": {"steps": 2, "weight": 10, "alpha": 1, "beta": 10},
        "selection": {"target_probability": 0.9999999, "max_answer_kl": 0.1},
        "runtime_limit_seconds": 120,
        "new_fits": 8,
        "data_sha256": {
            str(p.relative_to(data)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in data.rglob("*")
            if p.is_file()
        },
    }
    (data / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(run.GPT2Tokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(run.GPTJForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(run, "generate", generate)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "test CPU")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run", "--data", str(data), "--output", str(output)])
    run.main()
    assert fingerprint(model) == identity
    assert not hasattr(model, "peft_config")
    completion = json.loads((output / "completion.json").read_text())
    assert completion["weights_restored"] and completion["new_fits"] == 8
    fits = json.loads((output / "fits.json").read_text())
    assert all(r["roundtrip"]["logits_exact"] for r in fits if r["family"] == "lora")
    summary, decisions = summarize(output)
    assert len(summary["cases"]) == 8 and len(decisions) == 120
    original_support = next(
        r for r in summary["cases"] if r["case"] == 0 and r["condition"] == "consensus"
    )
    assert original_support["trainable_scalars"] == 64
    assert original_support["support_occurrences"] == 4
    assert original_support["unique_weight_coordinates"] == 48
    assert not any(g["advance_counts_pass"] for g in summary["groups"].values())
    from experiments.adapter_comparison.verify import check_results

    checked = check_results(output, inputs, protocol, cold, tokenizer)
    assert checked["sequences_checked"] == checked["decoded"] == 530
    assert checked["calibration_records_recomputed"] == 8
    from experiments.adapter_comparison.review import packet

    review = packet(output)
    assert review["observations"] == 530
    assert sum(len(g["occurrences"]) for g in review["items"]) == 530
    # Equal text for different prompts remains a separate review context.
    utility = [g for g in review["items"] if g["prompt"].startswith("utility")]
    assert len({g["prompt"] for g in utility}) == 32
    fits[0]["final"]["nll"] += 0.5
    (output / "fits.json").write_text(json.dumps(fits))
    with pytest.raises(AssertionError):
        check_results(output, inputs, protocol, cold)
