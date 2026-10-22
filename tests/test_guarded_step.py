from copy import deepcopy

import pytest
import torch

from experiments.guarded_training.step import Budget, Measurement, backtracked_step


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def initialized():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]).double())
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([2.0, -1.0]).double()
    optimizer.step()  # Include nonzero moments and an existing step counter.
    return parameter, optimizer


def test_rejected_large_step_then_accept_matches_one_independent_half_rate_adam_step():
    parameter, optimizer = initialized()
    initial = parameter.detach().clone()
    state = deepcopy(optimizer.state_dict())
    control = torch.nn.Parameter(initial.clone())
    reference = torch.optim.Adam([control], lr=0.1)
    reference.load_state_dict(deepcopy(state))
    control.grad = parameter.grad.clone()
    reference.param_groups[0]["lr"] = 0.05
    reference.step()
    reference.param_groups[0]["lr"] = 0.1
    calls = []
    rng = torch.random.get_rng_state()
    expected_draw = torch.rand(1)
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)

    def measure():
        calls.append(torch.rand(1))
        drift = float((parameter - initial).square().sum())
        return Measurement(0.9, drift, drift)

    result = backtracked_step(optimizer, measure, Measurement(1.0, 0, 0), Budget(0.01, 0.01))
    assert result["accepted"] and len(result["attempts"]) == 2
    assert result["attempts"][1]["scale"] == 0.5
    assert all(torch.equal(v, expected_draw) for v in calls)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    equal(parameter, control)
    equal(optimizer.state_dict(), reference.state_dict())


def test_exhaustion_restores_parameters_gradients_moments_rates_and_rng():
    parameter, optimizer = initialized()
    initial, gradient = parameter.detach().clone(), parameter.grad.clone()
    state, rng = deepcopy(optimizer.state_dict()), torch.random.get_rng_state()

    def reject():
        torch.rand(5)
        return Measurement(1.1, 0, 0)  # Safe drift cannot excuse a worse task loss.

    result = backtracked_step(optimizer, reject, Measurement(1.0, 0, 0), Budget(1, 1))
    assert not result["accepted"] and len(result["attempts"]) == 4
    equal(parameter, initial)
    equal(parameter.grad, gradient)
    equal(optimizer.state_dict(), state)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_measurement_exception_rolls_back_a_provisional_step():
    parameter, optimizer = initialized()
    initial, gradient = parameter.detach().clone(), parameter.grad.clone()
    state = deepcopy(optimizer.state_dict())

    def broken():
        parameter.grad.zero_()
        raise RuntimeError("measurement failed")

    with pytest.raises(RuntimeError, match="measurement failed"):
        backtracked_step(optimizer, broken, Measurement(1.0, 0, 0), Budget(1, 1))
    equal(parameter, initial)
    equal(parameter.grad, gradient)
    equal(optimizer.state_dict(), state)
