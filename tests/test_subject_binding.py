"""Check answer-position gradients, matched controls and calibration-only selection."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from experiments.fine_transfer.fit import fit, sequence
from experiments.subject_binding.analysis import select_weight
from experiments.subject_binding.fit import (
    answer_kl,
    capture_calibration,
    correction,
    fit_binding,
    prompt_ids,
)
from neuronpatch.editing import GPTJEditConfig
from neuronpatch.occurrence import OccurrenceColumnEdit


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        tokens = [7, 8] if text == " new answer" else [3, 4, 5]
        if "other" in text:
            tokens = [3, 9, 6, 2]
        return torch.tensor([tokens])


def model_and_config():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(721)
    model = (
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
    config = GPTJEditConfig(max_steps=3, early_stop_probability=None, delta_dtype=torch.float32)
    targets = [
        ("transformer.h.0.mlp.fc_out", 2),
        ("transformer.h.1.mlp.fc_out", 7),
        ("transformer.h.0.mlp.fc_out", 2),
    ]
    return model, config, targets


def test_answer_kl_uses_full_distribution_and_detaches_reference():
    reference = torch.tensor([0.3, 0.2, 0.5], requires_grad=True)
    logits = torch.tensor([1.0, 0.5, -0.5], requires_grad=True)
    loss = answer_kl(logits, reference)
    loss.backward()
    torch.testing.assert_close(
        logits.grad, logits.detach().softmax(-1) - reference.detach().softmax(-1)
    )
    assert reference.grad is None
    with pytest.raises(ValueError, match="vectors"):
        answer_kl(logits.unsqueeze(0), reference)


def test_zero_preservation_exactly_matches_fixed_step_transfer_control():
    base, config, targets = model_and_config()
    second = deepcopy(base)
    expected, history, updates = fit(
        base, Tokenizer(), ["Subject is"], "new answer", targets, config
    )
    actual, record, final = fit_binding(
        second, Tokenizer(), ["Subject is"], "new answer", targets, ["other is"], 0, config
    )
    assert updates == len(record) == 3
    torch.testing.assert_close(expected.delta, actual.delta, atol=0, rtol=0)
    assert [h["nll"] for h in record] == [h["nll"] for h in history]
    assert final["calibration_prompts"] == 1 and len(final["answer_kl"]) >= 1
    assert all(not m._forward_hooks for m in second.modules())


def test_sequential_preservation_backwards_match_joint_objective_updates():
    first, config, targets = model_and_config()
    second = deepcopy(first)
    original = {n: p.clone() for n, p in first.named_parameters()}
    prompts = ["Subject is"]
    calibration = ["other one", "Different subject"]
    weight = 10.0
    actual, _, _ = fit_binding(
        first, Tokenizer(), prompts, "new answer", targets, calibration, weight, config
    )
    examples = [sequence(Tokenizer(), p, "new answer", second.device) for p in prompts]
    references = []
    protected = []
    with torch.no_grad():
        for ids, labels in examples:
            references.append(
                second(input_ids=ids, use_cache=False).logits[0, : -len(labels) - 1].float().clone()
            )
    protected = capture_calibration(second, Tokenizer(), calibration)
    expected = OccurrenceColumnEdit(second, targets, delta_dtype=config.delta_dtype).eval()
    optimizer = torch.optim.Adam(expected.trainable_parameters(), lr=config.lr)
    for _ in range(config.max_steps):
        task, _ = correction(expected, examples, references, config)
        preservation = []
        for ids, start, reference in protected:
            logits = expected(input_ids=ids, use_cache=False).logits[0, start:].float()
            preservation.append(
                F.kl_div(logits.log_softmax(-1), reference.softmax(-1), reduction="none")
                .sum(-1)
                .mean()
            )
        (task + weight * torch.stack(preservation).mean()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(actual.delta, expected.delta, atol=1e-7, rtol=1e-5)
    for n, p in first.named_parameters():
        assert torch.equal(p, original[n]) and not p.requires_grad and p.grad is None
    assert all(not m._forward_hooks for m in first.modules())


def test_penalty_choice_requires_both_cases_and_ignores_evaluation_results():
    rows = [
        {
            "case": c,
            "weight": w,
            "final": {"nll": 0.05, "max_answer_kl": 0.05},
            "evaluation_hits": 0,
        }
        for w in (1, 10)
        for c in (0, 1)
    ]
    threshold = {"target_probability": 0.9, "max_answer_kl": 0.1}
    assert select_weight(rows, [0, 1, 10], [0, 1], threshold) == 1
    rows[1]["final"]["max_answer_kl"] = 0.2
    assert select_weight(rows, [0, 1, 10], [0, 1], threshold) == 10
    for r in rows:
        r["evaluation_hits"] = 999
    assert select_weight(rows, [0, 1, 10], [0, 1], threshold) == 10
    rows[-1]["final"]["nll"] = 1.0
    assert select_weight(rows, [0, 1, 10], [0, 1], threshold) is None
    with pytest.raises(ValueError, match="per case"):
        select_weight(rows[:-1], [0, 1, 10], [0, 1], threshold)


def test_complete_tiny_worker_replays_artifacts_and_restores_base(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    from experiments.fine_transfer.run import merge, restore, save_columns
    from experiments.fine_transfer.scope_analysis import digest_output
    from experiments.subject_binding import run
    from experiments.subject_binding.analysis import summarize
    from neuronpatch.patch import fingerprint

    model, config, targets = model_and_config()
    tokenizer = Tokenizer()
    before = fingerprint(model)
    data = tmp_path / "data"
    data.mkdir()

    @torch.no_grad()
    def generate(base, tok, prompt):
        ids = prompt_ids(tok, prompt, base.device)
        generated = base.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        tokens = generated[0, ids.shape[1] :].tolist()
        return {"text": str(tokens), "tokens": tokens, "truncated": False}

    cases, cold = [], []
    for case in (0, 1):
        edit = OccurrenceColumnEdit(model, targets, delta_dtype=torch.bfloat16)
        with torch.no_grad():
            edit.delta.copy_(torch.randn_like(edit.delta) * 0.01)
        edit.save(data / f"case-{case}-reference")
        probes = [{"prompt": f"reference{case}-{i}"} for i in (0, 1)]
        for p in probes:
            p["base"] = digest_output(generate(model, tokenizer, p["prompt"]))
        saved = save_columns(model, targets)
        try:
            merge(model, targets, edit.delta)
            for p in probes:
                p["edited"] = digest_output(generate(model, tokenizer, p["prompt"]))
        finally:
            restore(model, saved)
        cold.append({"case": case, "probes": probes})
        kinds = ["rewrite", "train/1", "train/2", "rephrase", "fresh_wording", "fresh_wording"]
        cases.append(
            {
                "case": case,
                "targets": [list(t) for t in targets],
                "trainable_scalars": len(targets) * 16,
                "target": "new answer",
                "train_prompts": ["Subject is"],
                "edit_probes": [
                    {"kind": k, "prompt": f"subject{case}-{i}"} for i, k in enumerate(kinds)
                ],
                **{
                    name: [
                        {
                            "source_index": 100 + i,
                            "subject": f"Other{case}-{name}-{i}",
                            "prompt": f"other {case} {name} {i}",
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
        "utility": [{"prompt": f"Utility{i} is", "gold": "unknown"} for i in range(32)],
    }
    for name, value in (("inputs.json", inputs), ("cold.json", cold)):
        (data / name).write_text(json.dumps(value))
    protocol = {
        "model": "tiny",
        "revision": "test",
        "base_fingerprint": before,
        "preservation_weights": [0.0, 1.0, 10.0],
        "runtime_limit_seconds": 60,
        "new_fits": 6,
        "selection": {"target_probability": 0.9999999, "max_answer_kl": 0.1},
        "fit": {"steps": 2, "lr": config.lr, "alpha": 1, "beta": 10, "gamma": 0.1},
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
    assert fingerprint(model) == before
    completion = json.loads((output / "completion.json").read_text())
    assert completion["new_fits"] == 6 and completion["weights_restored"]
    assert completion["selected_weight"] is None
    summary, decisions = summarize(output)
    assert len(summary["cases"]) == 6 and len(decisions) == 90
    assert not summary["advance_counts_pass"]
    from experiments.subject_binding.verify import check_results

    checks = check_results(output, inputs, protocol, cold)
    assert (
        checks["output_sequences_checked"] == 424 and checks["calibration_records_recomputed"] == 6
    )
    fits = json.loads((output / "fits.json").read_text())
    fits[0]["final"]["nll"] += 0.1
    (output / "fits.json").write_text(json.dumps(fits))
    with pytest.raises(AssertionError):
        check_results(output, inputs, protocol, cold)


def test_saved_calibration_metrics_average_prompts_equally_with_unequal_windows():
    from experiments.subject_binding.measure import statistics

    reference = torch.tensor([[0.4, 0.6], [0.3, 0.7], [0.8, 0.2]]).log()
    logits = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.2, 0.8]]).log()
    probabilities = torch.tensor([[-1.0, -3.0], [-0.5, -2.0]])
    actual = statistics(logits, reference, probabilities, torch.tensor([1, 2]))
    p = reference.double().softmax(-1)
    q = logits.double().softmax(-1)
    token_kl = (p * (p.log() - q.log())).sum(-1)
    assert actual["nll"] == 3.0
    assert actual["mean_answer_kl"] == pytest.approx(
        float((token_kl[0] + token_kl[1:].mean()) / 2), abs=1e-12
    )
    assert actual["max_answer_kl"] == pytest.approx(float(token_kl.max()), abs=1e-12)
