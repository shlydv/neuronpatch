from copy import deepcopy

from experiments.paired_mlp.analysis import choose_rates
from experiments.paired_mlp.data import whole_hit


def test_correct_start_cannot_hide_extra_content_or_failure_to_stop():
    answer = {"text": "Syria.", "ended": True, "unexpected_special": False}
    assert whole_hit(answer, ["Syria"])
    assert not whole_hit({**answer, "text": "Syria. Here is a made-up biography."}, ["Syria"])
    assert not whole_hit({**answer, "ended": False}, ["Syria"])
    assert not whole_hit({**answer, "unexpected_special": True}, ["Syria"])


def test_validation_rate_selection_ignores_test_success_and_rejects_preservation_loss():
    rows = []
    for rate in (0.0003, 0.001):
        for case in (0, 1):
            rows.append(
                {
                    "case": case,
                    "condition": {"family": "paired_mean", "seed": 43, "id": str(rate), "lr": rate},
                    "target_val_nll": 0.1 if rate == 0.0003 else 0.2,
                    "banks": {
                        "target_val": {"hits": 1},
                        "scope_val": {"lost": [], "new_target_mentions": []},
                        "utility_val": {"lost": [], "new_target_mentions": []},
                        "target_test": {"hits": 0 if rate == 0.0003 else 4},
                    },
                }
            )
    assert choose_rates(rows) == {"paired_mean-43": "0.0003"}
    changed = deepcopy(rows)
    for row in changed:
        row["banks"]["target_test"]["hits"] = 999
    assert choose_rates(changed) == choose_rates(rows)
    rows[0]["banks"]["utility_val"]["lost"] = ["protected-answer"]
    assert choose_rates(rows) == {"paired_mean-43": "0.001"}
    rows[2]["banks"]["scope_val"]["new_target_mentions"] = ["other-subject"]
    assert choose_rates(rows) == {"paired_mean-43": None}
