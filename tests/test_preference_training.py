import copy
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.instruction_boundary.preference_objective import dpo_loss
from experiments.instruction_boundary.preference_training import (
    PreparedPair,
    backward_pair,
    cache_references,
    fit_preferences,
    reference_lookup,
    score,
)
from neuronpatch.supervision import ResponseExample


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5)
        self.head = torch.nn.Linear(5, 8, bias=False)
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.calls += 1
        hidden = (self.embedding(input_ids) * attention_mask[:, :, None]).cumsum(1)
        return SimpleNamespace(logits=self.head(hidden))


def fixture():
    torch.manual_seed(13)
    model = TinyLM().eval().requires_grad_(False)

    def sample(ids):
        ids = torch.tensor([ids])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, 2:] = True
        return ResponseExample(ids, torch.ones_like(mask), mask, 2)

    a, b = sample([1, 3, 4, 2]), sample([1, 3, 5, 6, 2])
    pairs = [PreparedPair("p", "identity-p", a, b), PreparedPair("q", "identity-q", b, a)]
    cache = cache_references(model, pairs, base_identity="tiny-base")
    model.head.requires_grad_(True)
    return model, pairs, cache


def test_recomputed_gradient_matches_direct_dpo_with_unequal_response_lengths():
    model, pairs, cache = fixture()
    pair = pairs[0]
    ref = cache["records"][0]
    # Nonzero margin, not merely the initial policy==reference special case.
    with torch.no_grad():
        model.head.weight.add_(0.03 * torch.randn_like(model.head.weight))
    c, r = score(model, pair.chosen), score(model, pair.rejected)
    direct = dpo_loss(c, r, c.new_tensor(ref["chosen"]), c.new_tensor(ref["rejected"]), beta=0.2)
    (direct * 32).backward()
    expected = model.head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    before = model.calls
    result = backward_pair(model, pair, ref, objective="dpo", beta=0.2, scale=32)
    torch.testing.assert_close(model.head.weight.grad, expected, atol=2e-6, rtol=1e-5)
    assert result["loss"] == pytest.approx(direct.item())
    assert model.calls - before == result["forward_calls"] == 4
    assert model.embedding.weight.grad is None


def test_cache_mismatch_and_stochastic_mode_rejected():
    model, pairs, cache = fixture()
    with pytest.raises(ValueError, match="base"):
        reference_lookup(pairs, cache, base_identity="other")
    changed = copy.deepcopy(cache)
    changed["records"][0]["chosen"] += 1
    with pytest.raises(ValueError, match="content"):
        reference_lookup(pairs, changed, base_identity="tiny-base")
    with pytest.raises(ValueError, match="tokenized"):
        reference_lookup(
            [PreparedPair("p", "changed", pairs[0].chosen, pairs[0].rejected)],
            cache,
            base_identity="tiny-base",
        )
    model.train()
    with pytest.raises(ValueError, match="eval"):
        backward_pair(model, pairs[0], cache["records"][0], objective="dpo", beta=0.1, scale=1)
    model.eval()
    with pytest.raises(ValueError, match="frozen"):
        cache_references(model, pairs, base_identity="tiny-base")


@pytest.mark.parametrize("objective", ["sft", "dpo"])
def test_common_optimizer_matches_direct_objective_and_preservation(objective):
    model, pairs, cache = fixture()
    control = copy.deepcopy(model)
    preserved = [pairs[0].chosen]
    config = {
        "steps": 3,
        "lr": 0.001,
        "seed": 43,
        "beta": 0.1,
        "loss_scale": 32.0,
        "minimum_loss_scale": 1.0,
        "preservation_weight": 0.7,
        "gradient_clip": 1.0,
    }
    before = model.calls
    history = fit_preferences(
        model,
        pairs,
        preserved,
        cache,
        base_identity="tiny-base",
        objective=objective,
        protocol=config,
    )
    opt = torch.optim.Adam(control.head.parameters(), lr=0.001)
    for row in history:
        pair = pairs[row["example_indices"][0]]
        reference = reference_lookup(pairs, cache, base_identity="tiny-base")[pair.id]
        opt.zero_grad(set_to_none=True)
        c = score(control, pair.chosen)
        if objective == "sft":
            loss = -c / pair.chosen.completion_mask.sum()
        else:
            r = score(control, pair.rejected)
            loss = dpo_loss(
                c,
                r,
                c.new_tensor(reference["chosen"]),
                c.new_tensor(reference["rejected"]),
                beta=0.1,
            )
        preserve = -score(control, preserved[0]) / preserved[0].completion_mask.sum()
        (loss + 0.7 * preserve).backward()
        torch.nn.utils.clip_grad_norm_(control.head.parameters(), 1.0)
        opt.step()
        assert row["loss"] == pytest.approx(loss.item(), abs=1e-6)
        assert row["gradient_retries"] == []
    torch.testing.assert_close(model.head.weight, control.head.weight, atol=1e-7, rtol=1e-5)
    assert model.calls - before == 3 * (5 if objective == "dpo" else 2)
    assert all(math.isfinite(r["gradient_norm"]) for r in history)


def test_nonfinite_attempt_retries_same_pair_without_advancing_adam():
    model, pairs, cache = fixture()
    control = copy.deepcopy(model)
    config = {
        "steps": 1,
        "lr": 0.001,
        "seed": 43,
        "beta": 0.1,
        "loss_scale": 32.0,
        "minimum_loss_scale": 1.0,
        "preservation_weight": 0.7,
        "gradient_clip": 1.0,
    }
    calls = [0]

    def fail_once(gradient):
        calls[0] += 1
        return torch.full_like(gradient, float("inf")) if calls[0] == 1 else gradient

    handle = model.head.weight.register_hook(fail_once)
    history = fit_preferences(
        model,
        pairs,
        [pairs[0].chosen],
        cache,
        base_identity="tiny-base",
        objective="dpo",
        protocol=config,
    )
    handle.remove()
    expected = fit_preferences(
        control,
        pairs,
        [pairs[0].chosen],
        cache,
        base_identity="tiny-base",
        objective="dpo",
        protocol=config,
    )
    assert history[0]["loss_scale"] == 16.0
    assert len(history[0]["gradient_retries"]) == 1
    assert (
        history[0]["gradient_retries"][0]["pair_id"]
        == history[0]["pair_id"]
        == expected[0]["pair_id"]
    )
    torch.testing.assert_close(model.head.weight, control.head.weight, atol=1e-7, rtol=1e-5)


def test_tiny_qwen_fit_save_reload_and_base_restoration(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from experiments.paired_mlp.model import attach_lora, load_lora, unload_lora

    torch.manual_seed(19)
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    original = {k: v.clone() for k, v in model.state_dict().items()}
    _, pairs, _ = fixture()
    cache = cache_references(model, pairs, base_identity="qwen-fixture")
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ]
    fitted = attach_lora(model, names, rank=1, seed=43)
    config = {
        "steps": 2,
        "lr": 0.001,
        "seed": 43,
        "beta": 0.1,
        "loss_scale": 32.0,
        "minimum_loss_scale": 1.0,
        "preservation_weight": 0.7,
        "gradient_clip": 1.0,
    }
    history = fit_preferences(
        fitted,
        pairs,
        [pairs[0].chosen],
        cache,
        base_identity="qwen-fixture",
        objective="dpo",
        protocol=config,
    )
    assert len(history) == 2 and all(not r["gradient_retries"] for r in history)
    with torch.no_grad():
        live = score(fitted, pairs[0].chosen)
    trained = {k: p.detach().clone() for k, p in fitted.named_parameters() if p.requires_grad}
    directory = tmp_path / "adapter"
    fitted.save_pretrained(directory, safe_serialization=True)
    model = unload_lora(fitted).eval().requires_grad_(False)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    reloaded = load_lora(model, directory, names, rank=1, seed=43)
    assert all(torch.equal(v, dict(reloaded.named_parameters())[k]) for k, v in trained.items())
    with torch.no_grad():
        restored = score(reloaded, pairs[0].chosen)
    torch.testing.assert_close(live, restored, rtol=0, atol=0)
    model = unload_lora(reloaded)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


def test_four_arm_runner_uses_exported_models_and_restores_base(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from experiments.instruction_boundary.preference_experiment import run_comparison
    from neuronpatch.patch import fingerprint

    torch.manual_seed(29)
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    original = fingerprint(model)
    _, pairs, _ = fixture()
    visited = []

    def evaluate(fitted, directory):
        assert (directory / "adapter" / "adapter_model.safetensors").exists()
        assert not any(p.requires_grad for p in fitted.parameters())
        visited.append(directory.name)
        with torch.no_grad():
            return {"chosen_logp": float(score(fitted, pairs[0].chosen))}

    config = dict(
        steps=2,
        lr=0.001,
        seed=43,
        beta=0.1,
        loss_scale=32.0,
        minimum_loss_scale=1.0,
        preservation_weight=0.7,
        gradient_clip=1.0,
    )
    model, results = run_comparison(
        model,
        {"narrow": pairs[:1], "diverse": pairs},
        [pairs[0].chosen],
        names=["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"],
        rank=1,
        protocol=config,
        output=tmp_path / "run",
        evaluate=evaluate,
    )
    assert visited == ["narrow-sft", "narrow-dpo", "diverse-sft", "diverse-dpo"]
    assert fingerprint(model) == original
    assert (tmp_path / "run" / "complete.json").exists()
    assert all(r["successful_steps"] == 2 for r in results.values())
    assert results["narrow-sft"]["forward_calls"] == 4
    assert results["narrow-dpo"]["forward_calls"] == 10
    with pytest.raises(FileExistsError):
        run_comparison(
            model,
            {"narrow": pairs[:1], "diverse": pairs},
            [pairs[0].chosen],
            names=["model.layers.0.self_attn.q_proj"],
            rank=1,
            protocol=config,
            output=tmp_path / "run",
            evaluate=evaluate,
        )
