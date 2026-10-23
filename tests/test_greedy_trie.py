import pytest
import torch

from experiments.outcome_guard.decision_audit import greedy_trace


def test_unchanged_valid_mass_can_hide_a_wrong_greedy_root():
    prefixes, paths = [[], [0], [1]], [[0, 3], [1, 3]]
    suffix = torch.tensor([0.01, 0.01, 0.01, 0.97]).log()
    before = torch.stack([torch.tensor([0.58, 0.01, 0.40, 0.01]).log(), suffix, suffix])
    after = torch.stack([torch.tensor([0.295, 0.295, 0.40, 0.01]).log(), suffix, suffix])
    assert greedy_trace(before, prefixes, paths, 3)["accepted"]
    observed = greedy_trace(after, prefixes, paths, 3)
    assert not observed["accepted"] and observed["observed_tokens"] == [2]
    assert observed["trace"][0]["allowed_minus_outside_logit_gap"] < 0
    assert not observed["complete_response"]


def test_valid_alternative_and_late_invalid_continuation_are_distinguished():
    prefixes, paths = [[], [0], [0, 1]], [[0, 3], [0, 1, 3]]
    z = torch.tensor([[5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 0, 5.0]])
    alternate = greedy_trace(z, prefixes, paths, 3)
    assert alternate["accepted"] and alternate["observed_tokens"] == [0, 1, 3]
    z[-1] = torch.tensor([0, 0, 5.0, 0])
    bad = greedy_trace(z, prefixes, paths, 3)
    assert not bad["accepted"] and bad["observed_tokens"] == [0, 1, 2]
    assert len(bad["trace"]) == 3 and not bad["complete_response"]


def test_missing_prediction_context_is_rejected():
    with pytest.raises(ValueError, match="prediction prefix"):
        greedy_trace(torch.zeros(1, 4), [[]], [[0, 3]], 3)
