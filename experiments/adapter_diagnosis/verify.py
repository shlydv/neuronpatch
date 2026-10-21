"""Reconstruct every diagnostic metric and require exact parent control replay."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.adapter_comparison.verify import compare_metrics
from experiments.subject_binding.measure import statistics

from .run import verify_parent


def check_results(root, parent, protocol, tokenizer=None):
    def read(name):
        return json.loads((root / name).read_text())

    verify_parent(parent, protocol["parent_sha256"])
    done, identity, inputs = [read(n) for n in ("completion.json", "identity.json", "inputs.json")]
    assert identity["protocol"] == protocol
    assert identity["fingerprint"] == done["final_fingerprint"] == protocol["base_fingerprint"]
    assert done["status"] == "complete" and done["controls_exact"] and done["weights_restored"]
    assert done["new_fits"] == done["new_training_steps"] == 0
    assert inputs == json.loads((parent / "inputs.json").read_text())
    rows = read("interventions.json")
    variants = {r["id"]: r["keep"] for r in protocol["conditions"]}
    expected = {(c["case"], key) for c in inputs["cases"] for key in variants}
    assert len(rows) == len(expected) == done["interventions"]
    assert {(r["case"], r["condition"]) for r in rows} == expected
    selected = protocol["parent_condition"]
    checked, decoded, reconstructed, control_count = 0, 0, 0, 0
    bases = read("base-metrics.json")
    assert {r["case"] for r in bases} == {c["case"] for c in inputs["cases"]}
    for case in inputs["cases"]:
        c = case["case"]
        reference = load_file(str(root / f"case-{c}-calibration-base.safetensors"))
        previous = load_file(str(parent / f"case-{c}-calibration-base.safetensors"))
        assert reference.keys() == previous.keys()
        assert all(torch.equal(t, previous[k]) for k, t in reference.items())
        factors = load_file(str(parent / f"case-{c}-{selected}-edit/adapter_model.safetensors"))
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in factors.values())
        sizes = {}
        for key, tensor in factors.items():
            name, part = key.removeprefix("base_model.model.").rsplit(".lora_", 1)
            assert part in ("A.weight", "B.weight")
            sizes[name] = sizes.get(name, 0) + tensor.numel()
        assert set(sizes) == set(variants["full"])
        for name in ["base", *variants]:
            answer = read(f"case-{c}-{name}.json")
            assert answer["case"] == c
            if name in ("base", "full"):
                prior_name = "base" if name == "base" else selected
                prior = json.loads((parent / f"case-{c}-{prior_name}.json").read_text())
                assert answer == prior
                control_count += sum(len(v) for k, v in answer.items() if k != "case")
            for bank, probes in [
                ("edit", case["edit_probes"]),
                ("utility", inputs["utility"]),
                *((k, case[k]) for k in ("calibration", "evaluation", "exposed_scope")),
            ]:
                assert len(answer[bank]) == len(probes)
                for value in answer[bank]:
                    assert set(value) == {"text", "tokens", "truncated"}
                    assert isinstance(value["text"], str) and type(value["truncated"]) is bool
                    assert 0 < len(value["tokens"]) <= 40
                    assert all(type(t) is int and t >= 0 for t in value["tokens"])
                    checked += 1
                    if tokenizer is not None:
                        assert (
                            tokenizer.decode(value["tokens"], skip_special_tokens=True)
                            == value["text"]
                        )
                        decoded += 1
            tensors = load_file(str(root / f"case-{c}-{name}-calibration.safetensors"))
            assert set(tensors) == {"logits", "target_log_probabilities"}
            assert tensors["logits"].shape == reference["logits"].shape
            assert tensors["target_log_probabilities"].ndim == 2
            assert tensors["target_log_probabilities"].shape[0] == len(case["train_prompts"])
            assert all(
                t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values()
            )
            assert (tensors["target_log_probabilities"] <= 0).all()
            if tokenizer is not None:
                assert tensors["target_log_probabilities"].shape[1] == len(
                    tokenizer.encode(" " + case["target"], add_special_tokens=False)
                )
            metrics = statistics(
                tensors["logits"],
                reference["logits"],
                tensors["target_log_probabilities"],
                reference["lengths"],
            )
            if name == "base":
                assert torch.equal(tensors["logits"], reference["logits"])
                reported = next(r for r in bases if r["case"] == c)["metrics"]
            else:
                row = next(r for r in rows if r["case"] == c and r["condition"] == name)
                assert row["keep"] == variants[name] and row["base_restored"]
                assert set(row["excluded"]) == set(sizes) - set(variants[name])
                assert row["new_trainable_scalars"] == row["new_training_steps"] == 0
                assert row["retained_factor_scalars"] == sum(sizes[n] for n in variants[name])
                assert row["excluded_factor_scalars"] == sum(sizes[n] for n in row["excluded"])
                reported = row["metrics"]
            compare_metrics(metrics, reported)
            if name == "full":
                old = load_file(str(parent / f"case-{c}-{selected}-calibration.safetensors"))
                assert all(torch.equal(t, old[k]) for k, t in tensors.items())
            reconstructed += 1
    controls = read("controls.json")
    assert len(controls) == len(inputs["cases"])
    assert all(all(value is True for key, value in r.items() if key != "case") for r in controls)
    if "expected_observations" in protocol:
        assert checked == protocol["expected_observations"]
        assert control_count == protocol["expected_control_observations"]
    return {
        "status": "Execution/control/calibration verified; not a quality claim",
        "sequences_checked": checked,
        "decoded": decoded,
        "exact_control_observations": control_count,
        "calibration_records_recomputed": reconstructed,
        "new_fits": 0,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    tokenizer = None
    if args.decode:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(
            protocol["model"], revision=protocol["revision"], local_files_only=True
        )
    result = check_results(args.results, args.parent, protocol, tokenizer)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
