"""Decode all outputs and independently recompute complete-response losses/budgets."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from neuronpatch.supervision import chat_response

from .data import messages
from .run import BANKS


def verify(root, protocol, tokenizer):
    def read(name):
        return json.loads((root / name).read_text())

    inputs, identity, fits, completion = (
        read(n) for n in ("inputs.json", "identity.json", "fits.json", "completion.json")
    )
    assert (
        hashlib.sha256((root / "inputs.json").read_bytes()).hexdigest() == protocol["inputs_sha256"]
    )
    assert identity["protocol"] == protocol
    layers = protocol["architecture"]["layers"]
    hidden = protocol["architecture"]["hidden"]
    width = protocol["architecture"]["intermediate"]
    lora = 3 * protocol["rank"] * layers * (hidden + width)
    neurons = (lora + 3 * hidden // 2) // (3 * hidden)
    quotient, remainder = divmod(neurons, layers)
    layer_counts = [
        quotient + ((i + 1) * remainder // layers - i * remainder // layers) for i in range(layers)
    ]
    assert identity["layer_neuron_counts"] == layer_counts
    assert identity["budgets"] == {
        "lora": lora,
        "paired": 3 * hidden * neurons,
        "outgoing": 3 * hidden * neurons,
    }
    assert completion["status"] == "complete" and completion["base_restored"]
    assert completion["final_fingerprint"] == identity["base_fingerprint"]
    assert len(fits) == completion["new_fits"] == protocol["new_fits"]
    cases = {r["case"]: r for r in inputs["cases"]}
    expected = {(c, condition["id"]) for c in cases for condition in protocol["conditions"]}
    actual = {(r["case"], r["condition"]["id"]) for r in fits}
    assert actual == expected and len(actual) == len(fits)
    tokens = read("tokenization.json")
    # Tokenize every row first, also allowing deterministic tokenizer doubles in tests.
    for case in inputs["cases"]:
        for bank in BANKS:
            for row in case[bank]:
                example = chat_response(
                    tokenizer,
                    messages(row),
                    row["answers"][0],
                    max_length=protocol["max_sequence_tokens"],
                )
                assert tokens[row["id"]] == {
                    "ids": example.input_ids[0].tolist(),
                    "prompt_length": example.prompt_length,
                }
    decoded = 0
    for c, case in cases.items():
        if "discovery_indices" in protocol:
            discovery = read(f"case-{c}-discovery.json")
            assert discovery["training_indices"] == protocol["discovery_indices"]
            assert discovery["backward_passes"] == len(protocol["discovery_indices"])
            assert discovery["score"] == protocol["selection_score"]
            assert discovery["layer_counts"] == layer_counts
        for name in ["base", *[p["id"] for p in protocol["conditions"]]]:
            outputs = read(f"case-{c}-{name}.json")
            assert set(outputs) == set(BANKS)
            for bank in BANKS:
                assert len(outputs[bank]) == len(case[bank])
                for row, answer in zip(case[bank], outputs[bank], strict=True):
                    assert answer["id"] == row["id"]
                    new = answer["tokens"]
                    assert 0 < len(new) <= protocol["max_new_tokens"]
                    assert tokenizer.eos_token_id not in new[:-1]
                    assert answer["text"] == tokenizer.decode(new, skip_special_tokens=True)
                    ended = new[-1] == tokenizer.eos_token_id
                    assert answer["ended"] == ended and answer["truncated"] == (not ended)
                    assert ended or len(new) == protocol["max_new_tokens"]
                    special = any(
                        t in tokenizer.all_special_ids and t != tokenizer.eos_token_id for t in new
                    )
                    assert answer["unexpected_special"] == special
                    decoded += 1
    tensor_records = 0
    for fit in fits:
        c = fit["case"]
        condition = fit["condition"]
        name = condition["id"]
        case = cases[c]
        assert condition in protocol["conditions"]
        assert len(fit["history"]) == protocol["steps"]
        assert [r["step"] for r in fit["history"]] == list(range(1, protocol["steps"] + 1))
        generator = torch.Generator().manual_seed(protocol["seed"])
        orders = [
            torch.randperm(len(case[bank]), generator=generator).tolist()
            for bank in ("train", "preserve")
        ]
        assert [r["example_indices"] for r in fit["history"]] == [
            [order[step % len(order)] for order in orders]
            for step in range(protocol["steps"])
        ]
        assert all(
            math.isfinite(r[k])
            for r in fit["history"]
            for k in ("task_nll", "preserve_nll", "gradient_norm")
        )
        assert fit["reload_teacher_exact"] and fit["base_restored"]
        directory = root / f"case-{c}-{name}-adapter"
        if condition["family"] == "lora":
            values = load_file(str(directory / "adapter_model.safetensors"))
            config = json.loads((directory / "adapter_config.json").read_text())
            assert config["r"] == config["lora_alpha"] == protocol["rank"]
            assert config["lora_dropout"] == 0 and config["bias"] == "none"
            assert not config.get("use_dora", False) and not config.get("use_rslora", False)
            expected_shapes = {}
            for layer in range(layers):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    path = f"base_model.model.model.layers.{layer}.mlp.{projection}"
                    in_width, out_width = (
                        (width, hidden) if projection == "down_proj" else (hidden, width)
                    )
                    expected_shapes[f"{path}.lora_A.weight"] = (protocol["rank"], in_width)
                    expected_shapes[f"{path}.lora_B.weight"] = (out_width, protocol["rank"])
            assert set(values) == set(expected_shapes)
            assert all(tuple(values[key].shape) == shape for key, shape in expected_shapes.items())
            count = identity["budgets"]["lora"]
        else:
            values = load_file(str(directory / "patch.safetensors"))
            config = json.loads((directory / "patch.json").read_text())
            assert config["base_fingerprint"] == identity["base_fingerprint"]
            assert config["selections"] == fit["supports"]
            assert config["train_incoming"] == (condition["family"] != "outgoing_mean")
            assert len(fit["supports"]) == layers
            score = load_file(str(root / f"case-{c}-scores.safetensors"))
            generator = torch.Generator().manual_seed(condition["seed"])
            expected_shapes = {}
            for i, (selection, k) in enumerate(zip(fit["supports"], layer_counts, strict=True)):
                prefix = f"model.layers.{i}.mlp"
                incoming = [f"{prefix}.gate_proj", f"{prefix}.up_proj"]
                outgoing = f"{prefix}.down_proj"
                assert selection["incoming"] == incoming and selection["outgoing"] == outgoing
                if condition["family"] == "paired_random":
                    selected = torch.randperm(width, generator=generator)[:k].tolist()
                else:
                    key = (
                        "paired_empirical_fisher"
                        if condition["family"] == "paired_fisher"
                        else "outgoing_gradient_norm"
                        if condition["family"] == "outgoing_mean"
                        else "paired_gradient_norm"
                    )
                    if condition["family"] == "outgoing_mean":
                        k *= 3
                    selected = torch.argsort(score[f"{i}-{key}"], descending=True, stable=True)[
                        :k
                    ].tolist()
                assert selection["indices"] == selected and len(set(selected)) == k
                if config["train_incoming"]:
                    expected_shapes.update({name: (k, hidden) for name in incoming})
                expected_shapes[outgoing] = (hidden, k)
            assert set(values) == set(expected_shapes)
            assert all(tuple(values[key].shape) == shape for key, shape in expected_shapes.items())
            count = identity["budgets"]["paired"]
        assert sum(v.numel() for v in values.values()) == fit["trainable_scalars"] == count
        assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in values.values())
        tensor = load_file(str(root / f"case-{c}-{name}-teacher.safetensors"))
        measured = [*case["train"], *case["preserve"], *case["target_val"]]
        labels = []
        lengths = []
        for row in measured:
            sample = tokens[row["id"]]
            answer = sample["ids"][sample["prompt_length"] :]
            labels.extend(answer)
            lengths.append(len(answer))
        assert tensor["labels"].tolist() == labels and tensor["lengths"].tolist() == lengths
        assert tensor["logits"].shape[0] == len(labels)
        assert torch.isfinite(tensor["logits"]).all()
        loss = F.cross_entropy(tensor["logits"].double(), tensor["labels"], reduction="none")
        split = loss.split(lengths)
        task_count = len(case["train"])
        preserved_count = len(case["preserve"])
        end = task_count + preserved_count
        computed = {
            "task_mean_nll": sum(float(v.mean()) for v in split[:task_count]) / task_count,
            "task_worst_token_nll": max(float(v.max()) for v in split[:task_count]),
            "preserve_mean_nll": sum(float(v.mean()) for v in split[task_count:end])
            / preserved_count,
            "target_val_mean_nll": sum(float(v.mean()) for v in split[end:]) / len(split[end:]),
        }
        assert set(fit["metrics"]) == set(computed)
        assert all(
            math.isclose(value, fit["metrics"][key], rel_tol=1e-8, abs_tol=1e-8)
            for key, value in computed.items()
        )
        tensor_records += 1
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    return {
        "complete": True,
        "decoded": decoded,
        "fits": len(fits),
        "teacher_records_recomputed": tensor_records,
        "source_scope": "Artifact/control checks, not a quality or paper-superiority claim",
        "file_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    result = verify(args.results, protocol, tokenizer)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
