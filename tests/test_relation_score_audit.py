import pytest
import torch

from experiments.relation_learning.score_audit import summarize


def test_score_mass_is_not_squared_again_and_global_budget_can_move_layers():
    result = summarize(
        torch.tensor([[9.0, 8.0], [2.0, 1.0]]),
        {"balanced": [[0], [0]], "other": [[1], [1]]},
    )
    assert result["selected_neurons"] == 2
    assert result["global_score_layer_counts"] == [2, 0]
    assert result["global_score_fraction_total"] == pytest.approx(17 / 20)
    selected = result["recorded_supports"][0]
    assert selected["fraction_total"] == pytest.approx(11 / 20)
    assert selected["fraction_global_score_optimum"] == pytest.approx(11 / 17)


@pytest.mark.parametrize(
    "supports",
    [{"a": [[0, 0], []]}, {"a": [[2], []]}, {"a": [[0], []], "b": [[0], [1]]}],
)
def test_reject_invalid_or_unequal_supports(supports):
    with pytest.raises(ValueError):
        summarize(torch.ones(2, 2), supports)
