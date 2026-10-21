"""Frozen factor removal, exact merges, control replay and base restoration."""

import hashlib
import json

import pytest
import torch

from experiments.adapter_diagnosis.groups import conditions, retain_updates


def test_group_partition_and_final_layer_coverage():
    names = [
        f"transformer.h.{i}.{suffix}"
        for i in range(4)
        for suffix in [
            *(f"attn.{p}_proj" for p in ("q", "k", "v", "out")),
            "mlp.fc_in",
            "mlp.fc_out",
        ]
    ]
    variants = {r["id"]: set(r["keep"]) for r in conditions(names, 4)}
    assert len(variants) == 9 and variants["full"] == set(names)
    families = [variants[f"only-{key}"] for key in ("attention", "incoming", "outgoing")]
    assert len(set().union(*families)) == sum(map(len, families)) == 24
    for key in ("attention", "incoming", "outgoing", "final-three"):
        assert variants[f"without-{key}"] == set(names) - variants[f"only-{key}"]
    assert all("transformer.h.0." not in n for n in variants["only-final-three"])
    assert len(variants["only-final-three"]) == 18
    with pytest.raises(ValueError, match="all six"):
        conditions(names[:-1], 4)


def test_frozen_update_groups_and_complete_diagnostic(tmp_path, monkeypatch, capsys):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")

    from experiments.adapter_comparison.fit import capture_references
    from experiments.adapter_comparison.run import restore_weights, snapshot_weights
    from experiments.adapter_diagnosis import run
    from experiments.fine_transfer.lora_control import (
        attach_gptj_lora,
        gptj_targets,
        load_gptj_lora,
        merge_gptj_lora,
        remove_gptj_lora,
    )
    from experiments.subject_binding.fit import prompt_ids
    from experiments.subject_binding.measure import measure_merged
    from neuronpatch.patch import fingerprint

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            values = [7, 8] if text == " new answer" else [3, 4, 5]
            return torch.tensor([values]) if kwargs.get("return_tensors") == "pt" else values

        def decode(self, tokens, skip_special_tokens=True):
            return str(tokens)

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
        .half()
        .eval()
        .requires_grad_(False)
    )
    tokenizer = Tokenizer()

    @torch.no_grad()
    def generate(base, tok, prompt):
        ids = prompt_ids(tok, prompt, base.device)
        prediction = base.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )[0, ids.shape[1] :].tolist()
        return {"text": tok.decode(prediction), "tokens": prediction, "truncated": False}

    monkeypatch.setattr(run, "generate", generate)
    parent = tmp_path / "parent"
    parent.mkdir()

    def write(name, data):
        (parent / name).write_text(json.dumps(data))

    identity = fingerprint(model)
    names = gptj_targets(model, "attention_mlp")
    snapshot = snapshot_weights(model, names)
    inputs = {"cases": [], "utility": [{"prompt": "utility", "gold": "unknown"}]}
    for c in (0, 1):
        case = {
            "case": c,
            "target": "new answer",
            "train_prompts": ["Subject is"],
            "edit_probes": [{"kind": "rewrite", "prompt": "Subject is"}],
            **{
                k: [{"prompt": f"other {c}", "source_index": c, "reference_answers": "unknown"}]
                for k in ("calibration", "evaluation", "exposed_scope")
            },
        }
        inputs["cases"].append(case)

        def evaluate():
            return {
                "case": c,
                **{
                    bank: [generate(model, tokenizer, p["prompt"]) for p in probes]
                    for bank, probes in [
                        ("edit", case["edit_probes"]),
                        ("utility", inputs["utility"]),
                        *((k, case[k]) for k in ("calibration", "evaluation", "exposed_scope")),
                    ]
                },
            }

        write(f"case-{c}-base.json", evaluate())
        refs = capture_references(
            model,
            tokenizer,
            case["train_prompts"],
            case["target"],
            [p["prompt"] for p in case["calibration"]],
        )
        from safetensors.torch import save_file

        reference = {
            "logits": torch.cat([p[2].cpu() for p in refs.protected]),
            "lengths": torch.tensor([len(p[2]) for p in refs.protected]),
        }
        save_file(reference, parent / f"case-{c}-calibration-base.safetensors")
        adapted, _ = attach_gptj_lora(model)
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.02)
        artifact = parent / f"case-{c}-low-edit"
        adapted.save_pretrained(artifact, safe_serialization=True, save_embedding_layers=False)
        model = remove_gptj_lora(adapted)
        assert fingerprint(model) == identity
        # Independently check every retained merged projection against B@A.
        from safetensors.torch import load_file

        factors = load_file(str(artifact / "adapter_model.safetensors"))
        model = merge_gptj_lora(load_gptj_lora(model, artifact))
        full_weights = snapshot_weights(model, names)
        for name in names:
            a = factors[f"base_model.model.{name}.lora_A.weight"]
            b = factors[f"base_model.model.{name}.lora_B.weight"]
            delta = b @ a
            # PEFT0.11 adds in FP32 before restoring base dtype. Newer safe
            # merging can first round delta into the base dtype. Both are
            # official rounding orders, not cross-version bitwise parity.
            once = (snapshot[name].float() + delta).half()
            twice = snapshot[name] + delta.half()
            assert torch.equal(full_weights[name], once) or torch.equal(full_weights[name], twice)
        restore_weights(model, snapshot)
        for variant in conditions(names, 4):
            adapted = load_gptj_lora(model, artifact)
            result = retain_updates(adapted, names, variant["keep"])
            assert result["new_training_steps"] == result["new_trainable_scalars"] == 0
            assert result["retained_factor_scalars"] + result["excluded_factor_scalars"] == 1664
            model = merge_gptj_lora(adapted)
            for name in names:
                expected = full_weights[name] if name in variant["keep"] else snapshot[name]
                assert torch.equal(model.get_submodule(name).weight, expected)
            restore_weights(model, snapshot)
            assert fingerprint(model) == identity
        # Preserve a separately computed, fully merged parent control.
        model = merge_gptj_lora(load_gptj_lora(model, artifact))
        _, tensors = measure_merged(
            model, tokenizer, case["train_prompts"], case["target"], refs.protected, reference
        )
        save_file(tensors, parent / f"case-{c}-low-calibration.safetensors")
        write(f"case-{c}-low.json", evaluate())
        restore_weights(model, snapshot)
    write("inputs.json", inputs)
    write("identity.json", {"fingerprint": identity})
    write(
        "completion.json",
        {"status": "complete", "selected": {"lora": "low"}, "weights_restored": True},
    )
    protocol = {
        "parent_condition": "low",
        "base_fingerprint": identity,
        "conditions": conditions(names, 4),
        "new_fits": 0,
        "criteria": {
            "target_probability": 0.9,
            "max_answer_kl": 0.1,
            "minimum_untrained_prefixes": 0,
            "maximum_evaluation_leaks": 1,
        },
        "parent_sha256": {
            str(p.relative_to(parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in parent.rglob("*")
            if p.is_file()
        },
    }
    out = tmp_path / "output"
    out.mkdir()
    records = run.execute(model, tokenizer, parent, protocol, out, lambda: None)
    assert len(records) == 18 and all(r["base_restored"] for r in records)
    assert fingerprint(model) == identity
    done = json.loads((out / "completion.json").read_text())
    assert done["new_fits"] == done["new_training_steps"] == 0
    from experiments.adapter_diagnosis.verify import check_results

    verified = check_results(out, parent, protocol, tokenizer)
    assert verified["sequences_checked"] == verified["decoded"] == 100
    assert verified["exact_control_observations"] == 20
    assert verified["calibration_records_recomputed"] == 20
    from experiments.adapter_comparison.review import packet
    from experiments.adapter_diagnosis.analysis import summarize

    summary, decisions = summarize(out)
    assert len(summary["groups"]) == 9 and len(summary["cases"]) == 18
    assert len(decisions) == 54
    assert packet(out)["observations"] == 100
    row_file = out / "interventions.json"
    tampered = json.loads(row_file.read_text())
    tampered[0]["metrics"]["nll"] += 0.5
    row_file.write_text(json.dumps(tampered))
    with pytest.raises(AssertionError):
        check_results(out, parent, protocol, tokenizer)
    row_file.write_text(json.dumps(records))
    logs = capsys.readouterr().out
    assert logs.index("all_controls_replayed_before_ablation") < logs.index("0 only-attention")
    # A matching-hash but incompatible control must still stop before ablations.
    old = json.loads((parent / "case-1-base.json").read_text())
    old["utility"][0]["text"] = "bad cached control"
    write("case-1-base.json", old)
    protocol["parent_sha256"]["case-1-base.json"] = hashlib.sha256(
        (parent / "case-1-base.json").read_bytes()
    ).hexdigest()
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(AssertionError, match="Base generation replay differs"):
        run.execute(model, tokenizer, parent, protocol, bad, lambda: None)
    assert not any(bad.glob("*only-*.json"))
    assert fingerprint(model) == identity
