import json
from pathlib import Path

import pytest
import torch
from test_paired_worker import TinyTokenizer

from experiments.instruction_boundary import transfer_run
from experiments.instruction_boundary.prepare import build
from experiments.instruction_boundary.verify_transfer import verify
from neuronpatch.patch import fingerprint


def test_eval_decoder_recomputation_preserves_logits_and_gradients():
    from experiments.instruction_boundary.checkpointing import enable_decoder_recomputation

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(29)
    model = transformers.Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    parameters = [layer.mlp.gate_proj.weight for layer in model.model.layers]
    for parameter in parameters:
        parameter.requires_grad_(True)
    inputs = torch.arange(12).unsqueeze(0)
    original = fingerprint(model)

    def measure():
        output = model(inputs, use_cache=False).logits
        gradients = torch.autograd.grad(output.float().square().mean(), parameters)
        return output.detach(), gradients

    expected, gradients = measure()
    enable_decoder_recomputation(model)
    actual, recomputed = measure()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(gradients, recomputed, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in parameters)
    assert not any(module.training for module in model.modules())
    assert fingerprint(model) == original
    with torch.no_grad():
        torch.testing.assert_close(model(inputs, use_cache=False).logits, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="already installed"):
        enable_decoder_recomputation(model)


@pytest.mark.parametrize("recompute,matched", [(False, False), (True, False), (True, True)])
def test_full_transfer_training_selection_reload_merge_restore_on_tiny_half_model(
    tmp_path, monkeypatch, recompute, matched
):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.Qwen3Config(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(19)
    model = transformers.Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    original = fingerprint(model)

    class FixtureTokenizer(TinyTokenizer):
        def __len__(self):
            return 256

    tokenizer = FixtureTokenizer()
    data = build(
        [{"context": f"table {i}", "question": "value?", "ideal": "2"} for i in range(6)],
        {"train": 2, "calibration": 2, "confirmation": 2},
    )
    spec = json.loads(
        (
            Path(__file__).parents[1] / "experiments/instruction_boundary/transfer_protocol_v1.json"
        ).read_text()
    )
    spec.update(
        decoder_recomputation=recompute,
        steps=2,
        max_new_tokens=8,
        max_sequence_tokens=256,
        minimum_teachers={"train": 1, "calibration": 1},
    )
    if matched:
        spec["discovery_mode"] = "matched_authority"
        for arm in spec["arms"]:
            if arm["family"] == "probe_grouped":
                arm["family"] = "authority_grouped"
    spec["diagnostics"]["observer_steps"] = [1, 2]
    spec["grouped"]["selected_rows_per_projection"] = 4
    spec["expected_budgets"].update(lora_rank1_all_projections=512, grouped_two_groups_gate_up=128)
    calls = []
    evaluated_fingerprints = {}

    def deterministic_generation(model, tokenizer, rows, spec, path, check_time):
        # Only text sampling is stubbed. Real Qwen forwards, gradients, Adam,
        # PEFT/grouped adapters, serialization, reload and merge execute below.
        if "confirmation" in path.name and not path.name.startswith("base-"):
            assert (path.parent / "choices.json").exists()
            calls.append(path.name)
            evaluated_fingerprints[path.name] = fingerprint(model)
        records = []
        for row in rows:
            ids = transfer_run.prompt_inputs(tokenizer, row, model.device)["input_ids"][0].tolist()
            records.append(
                {
                    "id": row["id"],
                    "prompt_ids": ids,
                    "tokens": [*tokenizer.encode("answer"), 2],
                    "text": "answer",
                    "ended": True,
                }
            )
        transfer_run.write(path, records)
        return records

    monkeypatch.setattr(transfer_run, "generate", deterministic_generation)
    fits = transfer_run.execute(model, tokenizer, data, spec, tmp_path / "run", lambda: None)
    assert len(fits) == 8 and len(calls) == 4
    assert all(row["reload_probe_exact"] and row["base_restored"] for row in fits)
    assert {row["trainable_scalars"] for row in fits} == {128, 512}
    assert fingerprint(model) == original
    choices = json.loads((tmp_path / "run/choices.json").read_text())
    for choice in choices:
        row = next(r for r in fits if r["id"] == choice)
        peers = [r for r in fits if r["arm"] == row["arm"]]
        assert row == min(peers, key=lambda r: (r["validation"]["macro_nll"], r["lr"]))
    for row in fits:
        observed = json.loads((tmp_path / "run" / f"{row['id']}-observer.json").read_text())
        assert [r["step"] for r in observed] == [1, 2]
        assert all(set(r["banks"]) == {"task", "clean"} for r in observed)
    checked = verify(data, spec, tmp_path / "run", tokenizer, [2])
    assert checked["verified_fits"] == 8 and checked["verified_generations"] == 74
    if matched:
        import hashlib

        from experiments.instruction_boundary import replay as replay_module

        root = tmp_path / "run"
        pinned = [
            root / name
            for name in ("choices.json", "fits.json", "completion.json", "protocol.json")
        ]
        for choice in choices:
            pinned.extend(p for p in (root / f"{choice}-adapter").rglob("*") if p.is_file())
        manifest = {
            "choices": choices,
            "base_fingerprint": original,
            "files_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pinned
            },
        }
        assert replay_module.find_parent(tmp_path, manifest) == root
        monkeypatch.setattr(replay_module, "generate", deterministic_generation)
        receipts = replay_module.replay(
            model,
            tokenizer,
            root,
            manifest,
            data["splits"]["confirmation"],
            spec,
            root,
            lambda: None,
        )
        assert len(receipts) == 4
        assert all(r["new_training_steps"] == 0 for r in receipts)
        assert fingerprint(model) == original
        for choice in choices:
            reused = json.loads((root / f"reused-{choice}-confirmation.json").read_text())
            assert reused == json.loads((root / f"{choice}-confirmation.json").read_text())
            assert (
                evaluated_fingerprints[f"reused-{choice}-confirmation.json"]
                == evaluated_fingerprints[f"{choice}-confirmation.json"]
            )
        for name in ("teachers-train.json", "teachers-calibration.json", "supervision.json"):
            manifest["files_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        spec["reused_parent"] = manifest
        transfer_run.write(root / "protocol.json", spec)
        transfer_run.write(
            root / "full-comparison.json",
            {
                "status": "complete",
                "new_fits": 8,
                "reused": choices,
            },
        )
        assert verify(data, spec, root, tokenizer, [2])["verified_generations"] == 130
        receipts_path = root / "reused-comparators.json"
        receipts_path.write_text("[]")
        with pytest.raises(ValueError, match="replay incomplete"):
            verify(data, spec, root, tokenizer, [2])
        transfer_run.write(receipts_path, receipts)
        # Restore parent protocol before the independent adapter-corruption check.
        del spec["reused_parent"]
        transfer_run.write(root / "protocol.json", spec)
        adapter = next(p for p in pinned if p.suffix == ".safetensors")
        content = adapter.read_bytes()
        adapter.write_bytes(content + b"corrupted")
        with pytest.raises(ValueError, match="hash mismatch"):
            replay_module.verify_parent(root, manifest)
        adapter.write_bytes(content)
    path = tmp_path / "run/supervision.json"
    original_text = path.read_text()
    corrupt = json.loads(original_text)
    corrupt["train"]["task"][0]["target"] = "different target"
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="Teacher targets"):
        verify(data, spec, tmp_path / "run", tokenizer, [2])
    path.write_text(original_text)
    path = tmp_path / "run/choices.json"
    path.write_text(json.dumps(list(reversed(choices))))
    with pytest.raises(ValueError, match="Selection/completion"):
        verify(data, spec, tmp_path / "run", tokenizer, [2])
