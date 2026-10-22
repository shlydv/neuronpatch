import pytest
import torch
from torch.nn import functional as F

from experiments.functional_selection.logit_path import compare, moments


def test_exact_kl_matches_independent_definition_and_ignores_constant_logit_shift():
    base = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 0.0, 1.0]]).double()
    delta = torch.tensor([[0.1, 2.0, -1.0], [1.5, -0.3, 0.2]]).double()
    kl, quadratic = moments(base, delta)
    oracle = F.kl_div(
        (base + delta).log_softmax(-1),
        base.log_softmax(-1),
        log_target=True,
        reduction="none",
    ).sum(-1)
    torch.testing.assert_close(kl, oracle)
    shifted = moments(base + 40, delta - 80)
    torch.testing.assert_close(shifted[0], kl)
    torch.testing.assert_close(shifted[1], quadratic)


def test_linear_logit_path_can_have_severe_softmax_quadratic_error():
    base = torch.tensor([[10.0, 0.0]]).double()
    small = base + torch.tensor([[0.0, 10.0]]).double()
    large = base + torch.tensor([[0.0, 30.0]]).double()
    row = compare(base, small, large, [1], 3)
    assert row["small_kl"] > 100 * row["small_finite_logit_quadratic"]
    assert row["logit_bend_base_quadratic"] == 0
    assert row["large_kl_from_small_logit_secant"] == pytest.approx(row["large_kl"])
    assert row["small_kl_from_large_logit_secant"] == pytest.approx(row["small_kl"])


def test_curved_logit_path_is_distinct_from_softmax_curvature_and_examples_are_equal():
    base = torch.zeros(3, 2).double()
    small = base + torch.tensor([[0, 0.001], [0, 0.001], [0, 0.003]]).double()
    large = base + 9 * (small - base)
    row = compare(base, small, large, [2, 1], 3)
    first = compare(base[:2], small[:2], large[:2], [2], 3)
    second = compare(base[2:], small[2:], large[2:], [1], 3)
    assert row["small_kl"] == pytest.approx((first["small_kl"] + second["small_kl"]) / 2)
    assert row["logit_bend_base_quadratic"] > 0
    assert row["large_kl_from_small_logit_secant"] < row["large_kl"] / 4
    with pytest.raises(ValueError, match="partition"):
        compare(base, small, large, [2], 3)
