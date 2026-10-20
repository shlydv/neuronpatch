"""Offline nonlinear fitting/export check for the matched null-basis control."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from .null_lora import NullBasisLoRA
from .prototype import protected_span


def qwen_check():
    """Check the same scoped adapter on a tiny ordinary Qwen3, without downloads."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from neuronpatch.patch import fingerprint

    from .helpers import Examples
    from .preflight import Tokenizer
    from .run import capture_spans, restore_base

    torch.manual_seed(717)
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                num_hidden_layers=3,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=101,
                max_position_embeddings=64,
                _attn_implementation="eager",
                eos_token_id=0,
                pad_token_id=0,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    examples = Examples(Tokenizer(), torch.device("cpu"), generation_tokens=3)
    rows = [
        {
            "id": f"fixture-{i}",
            "article": str(i),
            "context": f"context {i}",
            "question": f"question {i}",
            "answer": "A",
            "answers": ["A"],
            "kind": "answerable",
        }
        for i in range(4)
    ]
    targets = [f"model.layers.{i}.mlp.up_proj" for i in range(3)]
    anchors = [{**row, "_target_ids": [13 + i, 0]} for i, row in enumerate(rows[:2])]
    original_hash = fingerprint(model)
    backups = {name: model.get_submodule(name).weight.detach().clone() for name in targets}
    spans, _ = capture_spans(model, targets, anchors, examples)

    @torch.no_grad()
    def logits(row):
        inputs = examples.inputs(row, training=True)
        return model(**{k: v for k, v in inputs.items() if k != "labels"}, use_cache=False).logits

    baseline = [logits(row) for row in anchors]
    adapter = NullBasisLoRA(model, spans, rank=2, alpha=4).eval()
    initial = float(examples.loss(adapter, rows[2]).detach())
    optimizer = torch.optim.Adam(adapter.trainable_parameters(), lr=0.01)
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = examples.loss(adapter, rows[2])
        loss.backward()
        optimizer.step()
    final = float(examples.loss(adapter, rows[2]).detach())
    assert torch.isfinite(loss) and final < initial
    with adapter.applied():
        live = [logits(row) for row in anchors]
        heldout_live = logits(rows[3])
    live_drift = max(float((after - before).abs().max()) for before, after in zip(baseline, live))
    assert live_drift < 1e-5
    assert fingerprint(model) == original_hash
    factors = adapter.export_factors()
    try:
        with torch.no_grad():
            for name, values in factors.items():
                model.get_submodule(name).weight.add_(values["b"] @ values["a"])
        merged_drift = max(
            float((logits(row) - before).abs().max()) for row, before in zip(anchors, baseline)
        )
        deployment_drift = float((logits(rows[3]) - heldout_live).abs().max())
        assert deployment_drift < 1e-5
    finally:
        restore_base(model, backups)
    assert fingerprint(model) == original_hash
    return {
        "scope": "tiny random Qwen3 integration on CPU/FP32; no quality or CUDA claim",
        "initial_ce": initial,
        "final_ce": final,
        "max_anchor_logit_drift_live": live_drift,
        "max_anchor_logit_drift_merged": merged_drift,
        "max_live_merged_logit_difference": deployment_drift,
        "base_during_training_and_after_restore": "exactly unchanged",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qwen", action="store_true", help="Also check tiny Qwen3/FP32 integration"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(717)
    model = nn.Sequential(nn.Linear(12, 9), nn.Tanh(), nn.Linear(9, 4)).double().eval()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    anchors = torch.randn(3, 12, dtype=torch.float64)
    training = torch.randn(20, 12, dtype=torch.float64)
    fresh = torch.randn(20, 12, dtype=torch.float64)
    span = protected_span(anchors)
    adapter = NullBasisLoRA(model, {"0": span}, rank=2, alpha=2).eval()
    basis = adapter.null_0
    teacher_delta = (
        torch.randn(9, 2, dtype=torch.float64)
        @ torch.randn(2, span.free_directions, dtype=torch.float64)
        @ basis.T
    ) * 0.04

    @torch.no_grad()
    def teacher(inputs):
        return model[2](
            torch.tanh(F.linear(inputs, model[0].weight + teacher_delta, model[0].bias))
        )

    labels, heldout_labels = teacher(training), teacher(fresh)
    baseline = model(anchors).detach().clone()
    assert torch.equal(adapter(training), model(training))
    initial = float(F.mse_loss(adapter(training), labels).detach())
    initial_fresh = float(F.mse_loss(adapter(fresh), heldout_labels).detach())
    optimizer = torch.optim.Adam(adapter.trainable_parameters(), lr=0.025)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(adapter(training), labels)
        loss.backward()
        optimizer.step()
    final = float(F.mse_loss(adapter(training), labels).detach())
    final_fresh = float(F.mse_loss(adapter(fresh), heldout_labels).detach())
    anchor_drift = float((adapter(anchors) - baseline).abs().max().detach())
    assert final < initial * 0.001 and final_fresh < initial_fresh * 0.01
    assert anchor_drift < 1e-12
    assert float((adapter.effective_weight(0) @ anchors.T).abs().max().detach()) < 1e-12
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    expected = adapter(fresh).detach().clone()
    try:
        with adapter.applied():
            raise RuntimeError("injected failure")
    except RuntimeError as error:
        assert str(error) == "injected failure"
    assert not model[0]._forward_hooks
    assert torch.equal(model(anchors), baseline)
    factors = adapter.export_factors()["0"]
    try:
        adapter(fresh)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Exported adapter remained active")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "factors.safetensors"
        save_file({key: value.contiguous() for key, value in factors.items()}, path)
        restored = load_file(path)
        artifact_bytes = path.stat().st_size
    delta = restored["b"] @ restored["a"]
    deployed = model[2](torch.tanh(F.linear(fresh, model[0].weight + delta, model[0].bias)))
    assert torch.allclose(deployed, expected, atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        model[0].weight.add_(delta)
    assert torch.allclose(model(fresh), expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(model(anchors), baseline, atol=1e-12, rtol=1e-12)
    full_span_rejected = False
    try:
        protected_span(torch.eye(12, dtype=torch.float64))
    except ValueError:
        full_span_rejected = True
    assert full_span_rejected
    result = {
        "status": "passed",
        "scope": "tiny FP64 nonlinear fixture; no LLM quality or low-precision claim",
        "torch": torch.__version__,
        "initial_mse": initial,
        "final_mse": final,
        "initial_fresh_mse": initial_fresh,
        "final_fresh_mse": final_fresh,
        "max_anchor_change": anchor_drift,
        "protected_rank": span.rank,
        "free_input_directions": span.free_directions,
        "adapter_rank": 2,
        "allocated_parameters": adapter.num_trainable_parameters,
        "equivalent_ordinary_lora_parameters": 2 * (12 + 9),
        "base_weights_during_training": "exactly unchanged",
        "export_reload_and_merge": "match within 1e-12",
        "deployment_artifact_bytes": artifact_bytes,
        "exception_hooks_removed": True,
        "full_span_rejected": full_span_rejected,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("null_lora.py", "check_null_lora.py", "prototype.py")
        },
    }
    if args.qwen:
        import peft
        import transformers

        result["qwen_integration"] = qwen_check()
        result["transformers"] = transformers.__version__
        result["peft"] = peft.__version__
        result["qwen_helper_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("helpers.py", "preflight.py", "run.py")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
