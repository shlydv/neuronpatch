import math

import pytest
import torch

from neuronpatch.losses import completion_nll


def test_causal_shift_full_answer_eos_equal_example_weights_and_padding():
    # IDs 1 and 2 are answer and EOS; a second answer uses just EOS.
    ids = torch.tensor([[0, 0, 1, 2, 0], [0, 0, 2, 0, 0]])
    response = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=torch.bool)
    attended = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.bool)
    logits = torch.full((2, 5, 3), float("nan"), dtype=torch.float64)
    logits[0, 1] = torch.tensor([0.1, 0.6, 0.3]).log()
    logits[0, 2] = torch.tensor([0.2, 0.3, 0.5]).log()
    logits[1, 1] = torch.tensor([0.1, 0.7, 0.2]).log()
    logits.requires_grad_()
    result = completion_nll(logits, ids, response, attended)
    expected = ((-math.log(0.6) - math.log(0.5)) / 2 - math.log(0.2)) / 2
    assert float(result.detach()) == pytest.approx(expected, abs=1e-7)
    result.backward()
    selected_predictions = torch.zeros_like(response)
    selected_predictions[:, :-1] = response[:, 1:]
    assert torch.count_nonzero(logits.grad[~selected_predictions]) == 0
    # Softmax gradient scales by response length and by number of examples.
    torch.testing.assert_close(logits.grad[0, 1], torch.tensor([0.1, -0.4, 0.3]).double() / 4)
    torch.testing.assert_close(logits.grad[1, 1], torch.tensor([0.1, 0.7, -0.8]).double() / 2)


def test_same_correct_answer_start_but_bad_stopping_is_penalized():
    ids = torch.tensor([[0, 1, 2]])
    response = torch.tensor([[False, True, True]])
    attention = torch.ones_like(response)
    good = torch.tensor([[[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.3, 0.3, 0.4]]]).log()
    bad = good.clone()
    bad[0, 1] = torch.tensor([0.9, 0.05, 0.05]).log()
    assert completion_nll(bad, ids, response, attention) > completion_nll(
        good, ids, response, attention
    )
    assert completion_nll(good.half(), ids, response, attention).dtype == torch.float32


@pytest.mark.parametrize("problem", ["padding", "first", "empty", "no_context", "target", "nan"])
def test_invalid_response_alignment_rejected(problem):
    ids = torch.tensor([[0, 1, 2, 0]])
    response = torch.tensor([[False, True, True, False]])
    attention = torch.tensor([[True, True, True, False]])
    logits = torch.zeros(1, 4, 3)
    if problem == "padding":
        response[0, 3] = True
    elif problem == "first":
        response[0, 0] = True
    elif problem == "empty":
        response.zero_()
    elif problem == "no_context":
        attention[0, 0] = False
    elif problem == "target":
        ids[0, 2] = 3
    else:
        logits[0, 1, 0] = float("nan")
    with pytest.raises(ValueError):
        completion_nll(logits, ids, response, attention)
