from copy import deepcopy

import pytest

from experiments.instruction_boundary.replay_plan import choices, parents


def test_multiple_parent_subsets_keep_order_and_reject_ambiguous_identity():
    first = {"choices": ["a", "b"], "replay_choices": ["b"], "base_fingerprint": "base"}
    second = {"choices": ["c"], "base_fingerprint": "base"}
    spec = {"reused_parents": [first, second]}
    assert [label for p in parents(spec) for label in choices(p)] == ["b", "c"]
    broken = deepcopy(spec)
    broken["reused_parents"][1]["base_fingerprint"] = "different"
    with pytest.raises(ValueError, match="base model"):
        parents(broken)
    with pytest.raises(ValueError, match="unique across"):
        parents({"reused_parents": [first, first]})
    with pytest.raises(ValueError, match="subset"):
        choices({**first, "replay_choices": ["unknown"]})
    with pytest.raises(ValueError, match="one replay"):
        parents({**spec, "reused_parent": first})
