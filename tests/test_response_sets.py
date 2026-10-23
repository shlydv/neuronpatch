"""Independent probability-tree and gradient checks, without an LLM."""

import math

import pytest
import torch

from experiments.guarded_training.response_sets import covered_paths, log_response_mass


def logits(values):
    return torch.tensor(values, dtype=torch.float64).log()


def test_disjoint_complete_aliases_share_prefix_without_double_counting():
    # 0 = first word, 1 = optional continuation, 2 = other, 3 = EOS.
    contexts = {
        (): logits([0.8, 0.05, 0.1, 0.05]),
        (0,): logits([0.1, 0.3, 0.1, 0.5]),
        (0, 1): logits([0.05, 0.05, 0.1, 0.8]),
    }
    actual = log_response_mass(contexts, [(0, 3), (0, 1, 3), (0, 3)], eos_token_id=3)
    assert actual.exp().item() == pytest.approx(0.8 * 0.5 + 0.8 * 0.3 * 0.8)


def test_valid_redistribution_vs_actual_loss_of_accepted_mass():
    # A probability tree with a branch deciding short vs long wording.
    # Its suffix-token KL is large, but the two accepted strings keep equal mass.
    def tree(branch):
        return {
            (): logits([0.9, 0.025, 0.025, 0.025, 0.025]),
            (0,): logits(branch),
            (0, 1): logits([0.025, 0.025, 0.025, 0.025, 0.9]),
        }

    paths = [(0, 4), (0, 1, 4)]
    base = tree([0.025, 0.1, 0.05, 0.025, 0.8])
    # Match accepted mass: .8 + .1*.9 = .17 + .8*.9 = .89.
    alias = tree([0.01, 0.8, 0.01, 0.01, 0.17])
    damaged = tree([0.025, 0.1, 0.75, 0.025, 0.1])

    def p(x):
        return log_response_mass(x, paths, eos_token_id=4).exp().item()
    assert p(base) == pytest.approx(0.801)
    assert p(alias) == pytest.approx(p(base))
    assert p(damaged) == pytest.approx(0.171)
    kl = (base[(0,)].exp() * (base[(0,)] - alias[(0,)])).sum()
    assert kl.item() > 1.0


def test_partial_contexts_are_explicit_and_never_fill_in_missing_aliases():
    contexts = {(): logits([0.4, 0.3, 0.3]), (0,): logits([0.1, 0.1, 0.8])}
    paths = [(0, 2), (1, 2)]
    present, missing = covered_paths(contexts, paths, eos_token_id=2)
    assert present == ((0, 2),) and missing == ((1, 2),)
    assert log_response_mass(contexts, present, eos_token_id=2).exp().item() == pytest.approx(
        0.32
    )
    with pytest.raises(ValueError, match="missing prediction prefix"):
        log_response_mass(contexts, paths, eos_token_id=2)


def test_gradient_is_marginal_likelihood_not_average_reference_loss():
    root = torch.tensor([0.7, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    contexts = {
        (): root,
        (0,): logits([0.1, 0.1, 0.8]),
        (1,): logits([0.2, 0.2, 0.6]),
    }
    loss = -log_response_mass(contexts, [(0, 2), (1, 2)], eos_token_id=2)
    gradient = torch.autograd.grad(loss, root)[0]
    p = root.detach().softmax(-1)
    mass = p[0] * 0.8 + p[1] * 0.6
    posterior = torch.tensor([p[0] * 0.8 / mass, p[1] * 0.6 / mass, 0])
    torch.testing.assert_close(gradient, p - posterior)
    assert loss.item() == pytest.approx(-math.log(mass.item()))
    assert torch.autograd.gradcheck(
        lambda z: log_response_mass({**contexts, (): z}, [(0, 2), (1, 2)], eos_token_id=2),
        (root,),
    )


@pytest.mark.parametrize("paths", [[], [(0,)], [(0, 2, 1, 2)], [(2,)], [(True, 2)]])
def test_malformed_paths_rejected(paths):
    with pytest.raises(ValueError):
        log_response_mass({(): torch.zeros(3)}, paths, eos_token_id=2)


def test_malformed_contexts_rejected():
    for contexts in (
        {(): torch.tensor([0, 1, 2]), (0,): torch.zeros(3)},
        {(): torch.tensor([0.0, float("nan"), 0.0]), (0,): torch.zeros(3)},
        {(): torch.zeros(3), (0,): torch.zeros(4)},
        {(): torch.zeros(3), (0,): torch.zeros(3, dtype=torch.float64)},
    ):
        with pytest.raises(ValueError):
            log_response_mass(contexts, [(0, 2)], eos_token_id=2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_accumulate_in_float32(dtype):
    contexts = {
        (): torch.tensor([4.1, -1.7, 0.2], dtype=dtype, requires_grad=True),
        (0,): torch.tensor([-1.2, 0.1, 3.3], dtype=dtype, requires_grad=True),
    }
    actual = log_response_mass(contexts, [(0, 2)], eos_token_id=2)
    expected = contexts[()][0].float() - contexts[()].float().logsumexp(0)
    expected += contexts[(0,)][2].float() - contexts[(0,)].float().logsumexp(0)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(torch.isfinite(z.grad).all() for z in contexts.values())
