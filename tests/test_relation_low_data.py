import hashlib
import json
from collections import Counter

from experiments.paired_mlp.run import BANKS
from experiments.relation_learning.low_data import FRESH_WORLDS, prepare
from experiments.relation_learning.prepare import WORLDS, build


def test_original_frozen_data_is_unchanged_by_world_parameter():
    raw = (json.dumps(build(), indent=2) + "\n").encode()
    assert hashlib.sha256(raw).hexdigest() == (
        "b137b5433d69acbeb0c5a245f804c6bd6ace7f1161e58ad277c97fa21b4493bf"
    )


def test_low_data_has_no_extra_discovery_labels_and_fresh_confirmation_names():
    data, spec, raw = prepare()
    case = data["cases"][0]
    original = build()["cases"][0]
    assert spec["inputs_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(case["train"]) == 8 and case["preserve"] == original["preserve"]
    assert all(row in original["train"] for row in case["train"])
    assert spec["discovery_indices"] == list(range(8))
    assert Counter((r["family"], r["condition"]) for r in case["train"]) == {
        (family, condition): 1
        for family in ("interview", "release")
        for condition in ("missing", "target_only", "target_first", "target_last")
    }
    assert len(case["target_test"]) == 128
    assert Counter(r["variant"] for r in case["target_test"]) == {
        "original": 64,
        "transfer": 64,
    }
    assert len(case["scope_test"]) == 16 and len(case["utility_test"]) == 8
    old_names = {name for world in WORLDS for name in world}
    new_names = [name for world in FRESH_WORLDS for name in world]
    assert len(set(new_names)) == len(new_names) and old_names.isdisjoint(new_names)
    assert {r["world"] for r in case["target_test"]} == {10, 11, 12, 13}
    assert {r["world"] for r in case["scope_test"]} == {14}
    ids = [row["id"] for bank in BANKS for row in case[bank]]
    assert len(ids) == len(set(ids))
    assert spec["steps"] == 64 and spec["new_fits"] == len(spec["conditions"]) == 8
    assert spec["minimum_loss_scale"] == 1 / 128
    assert len({c["lr"] for c in spec["conditions"]}) == 2
