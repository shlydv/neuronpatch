import pytest
import torch

from experiments.paired_checkpoints.step_analysis import decompose, token_transition


def recorded_step(previous, current, correction=None, *, dtype=torch.float64):
    parameter = torch.nn.Parameter(torch.tensor(previous, dtype=dtype))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    saved = optimizer.state[parameter]
    state = {
        "parameter-before/0": parameter.detach().clone(),
        "adam-first-moment/0": saved["exp_avg"].clone(),
        "adam-second-moment/0": saved["exp_avg_sq"].clone(),
        "clipped-joint-gradient/0": torch.tensor(current, dtype=dtype),
        "correction-gradient/0": torch.tensor(
            current if correction is None else correction, dtype=dtype
        ),
    }
    parameter.grad = state["clipped-joint-gradient/0"].clone()
    optimizer.step()
    state["parameter-after/0"] = parameter.detach().clone()
    audit = {
        "step": 2,
        "parameter_names": ["delta"],
        "optimizer_group": {
            key: optimizer.param_groups[0][key] for key in ("lr", "betas", "eps", "weight_decay")
        },
    }
    return state, audit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_historical_term_can_overwhelm_current_descent_without_changing_adam_state(dtype):
    state, audit = recorded_step([0.5, -0.3], [-0.001, -0.002], dtype=dtype)
    before = {key: value.clone() for key, value in state.items()}
    result = decompose(state, audit)["totals"]
    assert result["joint_dot_current"] < 0
    assert result["joint_dot_history"] > 0
    assert result["joint_dot_actual"] > 0
    assert result["joint_cosine_actual"] > 0
    tolerance = 1e-7 if dtype == torch.float32 else 1e-15
    assert result["max_parameter_reconstruction_error"] < tolerance
    assert all(torch.equal(state[key], value) for key, value in before.items())


def test_correction_conflict_is_distinct_from_historical_momentum():
    state, audit = recorded_step([0.5, -0.3], [0.1, 0.2], correction=[-0.2, -0.1])
    result = decompose(state, audit)["totals"]
    assert result["joint_dot_current"] < 0
    assert result["joint_dot_actual"] < 0
    assert result["correction_dot_current"] > 0
    assert result["correction_dot_actual"] > 0


def test_zero_gradient_has_no_direction_even_while_adam_still_moves():
    state, audit = recorded_step([0.5], [0.0])
    result = decompose(state, audit)["totals"]
    assert result["actual_squared_norm"] > 0
    assert result["current_squared_norm"] == 0
    assert result["joint_cosine_actual"] is None


def test_corrupt_recorded_step_is_rejected():
    state, audit = recorded_step([0.5], [-0.001])
    state["parameter-after/0"] += 0.01
    with pytest.raises(AssertionError):
        decompose(state, audit)


def test_token_transition_uses_sample_offset_and_rejects_different_target_context():
    before = {
        "lengths": torch.tensor([2]),
        "labels": torch.tensor([0, 2]),
        "logits": torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
    }
    after = {
        "lengths": torch.tensor([1, 2]),
        "labels": torch.tensor([1, 0, 2]),
        "logits": torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
    }
    transition = token_transition(before, after, 1)
    assert transition["eos_token_id"] == 2
    first = transition["tokens"][0]
    assert first["before"]["top_token_id"] == 0
    assert first["after"]["top_token_id"] == 2
    assert first["after"]["target_probability"] == pytest.approx(
        1 / (2 + torch.exp(torch.tensor(2.0)))
    )
    after["labels"][1] = 1
    with pytest.raises(AssertionError):
        token_transition(before, after, 1)
