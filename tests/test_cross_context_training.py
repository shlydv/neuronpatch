import copy

import pytest
import torch
from test_preference_training import fixture

from experiments.instruction_boundary.preference_training import fit_preferences, score
from experiments.instruction_boundary.response_retention import cache_responses, selected_logits


def setup():
    model, pairs, cache = fixture()
    model.requires_grad_(False)
    clean = copy.deepcopy(pairs[0].chosen)
    clean.input_ids[0, 0] = 7  # Different teacher context, identical response.
    retained = cache_responses(model, [clean], base_identity="tiny-base", max_cache_bytes=10000)
    model.head.requires_grad_(True)
    config = dict(
        steps=4,
        lr=0.001,
        seed=43,
        beta=0.1,
        loss_scale=32.0,
        minimum_loss_scale=1.0,
        preservation_weight=0.7,
        gradient_clip=1.0,
    )
    return model, pairs, cache, clean, retained, config


def test_mixed_distillation_and_trusted_ce_matches_direct_adam():
    model, pairs, cache, clean, retained, config = setup()
    control = copy.deepcopy(model)
    history = fit_preferences(
        model,
        pairs,
        [clean],
        cache,
        base_identity="tiny-base",
        objective="cross_context",
        protocol=config,
        preservation_objective="kl",
        retained_responses=retained,
        task_teachers={"p": retained[0], "q": None},
    )
    optimizer = torch.optim.Adam([control.head.weight], lr=config["lr"])
    q = retained[0].reference_logits.double().softmax(-1)

    def kl(sample):
        logp = selected_logits(control, sample).double().log_softmax(-1)
        return (q * (q.log() - logp)).sum(-1).mean()

    seen = set()
    for row in history:
        pair = pairs[row["example_indices"][0]]
        seen.add(pair.id)
        optimizer.zero_grad()
        loss = (
            kl(pair.chosen)
            if pair.id == "p"
            else -score(control, pair.chosen) / pair.chosen.completion_mask.sum()
        )
        total = loss + config["preservation_weight"] * kl(clean)
        total.backward()
        torch.nn.utils.clip_grad_norm_([control.head.weight], config["gradient_clip"])
        optimizer.step()
        assert row["forward_calls"] == 2
        assert row["loss"] == pytest.approx(float(loss.detach()), abs=2e-6)
        assert row["task_target"] == ("clean_distribution" if pair.id == "p" else "hard_response")
    assert seen == {"p", "q"}
    torch.testing.assert_close(model.head.weight, control.head.weight, atol=2e-7, rtol=1e-6)
    assert model.embedding.weight.grad is None


def test_invalid_teacher_mapping_fails_before_update():
    model, pairs, cache, clean, retained, config = setup()
    before = copy.deepcopy(model.state_dict())
    for mapping in (
        {"p": retained[0]},
        {"p": None, "q": None},
        {"p": retained[0], "q": retained[0]},
    ):
        with pytest.raises(ValueError):
            fit_preferences(
                model,
                pairs,
                [clean],
                cache,
                base_identity="tiny-base",
                objective="cross_context",
                protocol=config,
                task_teachers=mapping,
            )
        for name, value in model.state_dict().items():
            assert torch.equal(value, before[name])
