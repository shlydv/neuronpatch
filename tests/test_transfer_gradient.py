"""Verify the proposed selector against full autograd column gradients."""

import pytest
import torch
from torch.nn import functional as F

from experiments.fine_transfer.gradient import agreement_scores, capture_gradient_factors


def test_gram_score_matches_materialized_autograd_with_multiple_token_positions():
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(7, 11, generator=generator, dtype=torch.float64, requires_grad=True)
    factors, gradients = [], []
    for length in (3, 5, 4):
        inputs = torch.randn(length, 11, generator=generator, dtype=torch.float64)
        derivatives = torch.randn(length, 7, generator=generator, dtype=torch.float64)
        loss = ((inputs @ weight.T) * derivatives).sum()
        gradients.append(torch.autograd.grad(loss, weight)[0])
        factors.append((inputs, derivatives))
    stacked = torch.stack(gradients)
    average = stacked.mean(0)
    expected = (stacked * average).sum(1).min(0).values / average.norm(dim=0)
    torch.testing.assert_close(agreement_scores(factors), expected, atol=1e-12, rtol=1e-12)


def test_opposed_prompt_gradients_are_not_mistaken_for_agreement():
    inputs = torch.tensor([[1.0, 2.0]])
    derivatives = torch.tensor([[1.0, 3.0]])
    assert torch.equal(
        agreement_scores([(inputs, derivatives), (inputs, -derivatives)]), torch.zeros(2).double()
    )
    with pytest.raises(ValueError, match="Finite"):
        agreement_scores([(inputs * float("nan"), derivatives)])


def test_arithmetic_prefix_does_not_accept_different_decimal_or_thousands_value():
    from experiments.fine_transfer.analyze import prefix_hit

    assert prefix_hit("4.0 is the result.", "4")
    assert prefix_hit("4\n", "4")
    assert not prefix_hit("4.5", "4")
    assert not prefix_hit("4,000", "4")
    assert not prefix_hit("4th", "4")
    assert not prefix_hit("4e2", "4")


def test_equal_retention_totals_cannot_hide_different_utility_damage():
    from experiments.fine_transfer.analyze import advancement

    control = {"held_out_wording": 4, "utility_losses": [[0, 2]]}
    candidate = {"original": 2, "held_out_wording": 6, "utility_losses": [[0, 3]]}
    groups = {
        "fine_augmented": control,
        "gradient_single": control,
        "gradient_consensus": candidate,
    }
    passed, comparisons = advancement(groups, 20)
    assert not passed
    assert comparisons["fine_augmented"]["new_utility_losses"] == [(0, 3)]
    candidate["utility_losses"] = [[0, 2]]
    assert advancement(groups, 20)[0]


def test_scope_bank_uses_exact_templates_and_never_filters_on_model_answers():
    from experiments.fine_transfer.prepare_scope import freeze

    rows = [
        {"subject": "Alice", "prompt": "Citizenship of Alice is", "target_new": "Newland"},
        {"subject": "Prize", "prompt": "Country associated with Prize is", "target_new": "Newland"},
    ] + [{} for _ in range(126)]
    rows += [
        {
            "subject": name,
            "prompt": f"Citizenship of {name} is",
            "ground_truth": [[gold]],
            "model_answer": "This unused field must not affect selection",
        }
        for name, gold in [("Alice", "Oldland"), ("Bob", "Newland")]
    ]
    rows += [
        {
            "subject": f"Person{i}",
            "prompt": f"Citizenship of Person{i} is",
            "ground_truth": [["Oldland"]],
        }
        for i in range(10)
    ]
    rows += [
        {
            "subject": f"OtherPrize{i}",
            "prompt": f"Country associated with OtherPrize{i} is",
            "ground_truth": [["Oldland"]],
        }
        for i in range(10)
    ]
    banks = freeze(rows)
    assert [r["source_index"] for r in banks[0]["probes"]] == list(range(130, 138))
    assert [r["source_index"] for r in banks[1]["probes"]] == list(range(140, 148))


def test_random_control_matches_layers_and_duplicate_pattern_without_changing_rng():
    from collections import Counter

    from experiments.fine_transfer.run import matched_random

    model = torch.nn.Module()
    model.first = torch.nn.Linear(11, 7)
    model.second = torch.nn.Linear(13, 7)
    targets = [("first", 2), ("second", 3), ("first", 2), ("first", 5)]
    before = torch.random.get_rng_state().clone()
    selected = matched_random(targets, model, 42)
    assert torch.equal(before, torch.random.get_rng_state())
    assert selected == matched_random(targets, model, 42)
    assert [name for name, _ in selected] == [name for name, _ in targets]
    assert selected[0] == selected[2] and selected[0] != selected[3]
    assert sorted(Counter(selected).values()) == sorted(Counter(targets).values())


def test_real_gptj_capture_matches_full_weight_gradient_and_removes_hooks():
    from transformers import GPTJConfig, GPTJForCausalLM

    torch.manual_seed(31)
    model = (
        GPTJForCausalLM(
            GPTJConfig(
                n_layer=2,
                n_embd=16,
                n_head=2,
                n_inner=23,
                rotary_dim=8,
                vocab_size=19,
                n_positions=32,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    ids = torch.tensor([[0, 3, 8, 7, 11]])
    names = [f"transformer.h.{layer}.mlp.fc_out" for layer in range(2)]
    factors, _ = capture_gradient_factors(model, ids, 3, 11, names)
    assert all(not m._forward_hooks for m in model.modules())
    # Independently materialize actual parameter gradients on this tiny model.
    for name in names:
        model.get_submodule(name).weight.requires_grad_(True)
    logits = model(input_ids=ids, use_cache=False).logits
    F.cross_entropy(logits[0, 3].unsqueeze(0), torch.tensor([11])).backward()
    for name in names:
        inputs, derivatives = factors[name]
        torch.testing.assert_close(
            derivatives.T @ inputs,
            model.get_submodule(name).weight.grad,
            atol=1e-7,
            rtol=1e-5,
        )


def test_single_prompt_training_reduces_exactly_to_the_verified_fine_workflow():
    from copy import deepcopy

    from transformers import GPTJConfig, GPTJForCausalLM

    from experiments.fine_transfer.fit import fit
    from neuronpatch.editing import GPTJEditConfig, fit_gptj_edit

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return torch.tensor([[7, 8]]) if text == " new answer" else torch.tensor([[3, 4, 5]])

    torch.manual_seed(41)
    first = GPTJForCausalLM(
        GPTJConfig(
            n_layer=4,
            n_embd=16,
            n_head=2,
            n_inner=24,
            rotary_dim=8,
            vocab_size=19,
            n_positions=32,
        )
    ).eval()
    second = deepcopy(first)
    config = GPTJEditConfig(
        max_steps=3,
        neurons_per_token=2,
        early_stop_probability=None,
        delta_dtype=torch.float32,
        score_dtype=torch.float32,
    )
    reference = fit_gptj_edit(first, Tokenizer(), "Subject is", "new answer", config)
    actual, history, updates = fit(
        second, Tokenizer(), ["Subject is"], "new answer", reference.edit.targets, config
    )
    assert updates == reference.updates == 3
    torch.testing.assert_close(actual.delta, reference.edit.delta, atol=0, rtol=0)
    assert [row["nll"] for row in history] == [row["nll"] for row in reference.history]
