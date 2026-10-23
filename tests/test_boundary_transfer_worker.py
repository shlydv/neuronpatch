import json
from pathlib import Path

import pytest
import torch
from test_paired_worker import TinyTokenizer

from experiments.instruction_boundary import transfer_run
from experiments.instruction_boundary.prepare import build
from experiments.instruction_boundary.verify_transfer import verify
from neuronpatch.patch import fingerprint


def test_full_transfer_training_selection_reload_merge_restore_on_tiny_half_model(
    tmp_path, monkeypatch
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
        steps=2,
        max_new_tokens=8,
        max_sequence_tokens=256,
        minimum_teachers={"train": 1, "calibration": 1},
    )
    spec["diagnostics"]["observer_steps"] = [1, 2]
    spec["grouped"]["selected_rows_per_projection"] = 4
    spec["expected_budgets"].update(lora_rank1_all_projections=512, grouped_two_groups_gate_up=128)
    calls = []

    def deterministic_generation(model, tokenizer, rows, spec, path, check_time):
        # Only text sampling is stubbed. Real Qwen forwards, gradients, Adam,
        # PEFT/grouped adapters, serialization, reload and merge execute below.
        if "confirmation" in path.name and not path.name.startswith("base-"):
            assert (path.parent / "choices.json").exists()
            calls.append(path.name)
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
