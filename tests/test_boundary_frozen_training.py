import hashlib
import json

import pytest

from experiments.instruction_boundary.frozen_training import (
    fixed_supports,
    load_assets,
    verify_schedule,
)


def test_frozen_assets_reject_corruption(tmp_path):
    files = [
        "supports.json",
        "teachers-train.json",
        "teachers-calibration.json",
        "parent-fits.json",
    ]
    hashes = {}
    for name in files:
        raw = b"{}"
        (tmp_path / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    assert set(load_assets(tmp_path, {"files_sha256": hashes})) == set(files)
    (tmp_path / "supports.json").write_text(json.dumps({"tampered": True}))
    with pytest.raises(ValueError, match="differs"):
        load_assets(tmp_path, {"files_sha256": hashes})


def test_fixed_supports_retain_identity_and_reject_learned_parameters():
    support = {"gate": {"indices": [4, 1, 3], "groups": [0, 1, 0]}}
    assets = {"supports.json": {"authority_grouped-43": support}}
    arms = [{"family": "authority_grouped", "seed": 43}, {"family": "lora", "seed": 43}]
    result = fixed_supports(assets, arms, {"gate": 5}, rows=3, groups=2)
    assert result["authority_grouped-43"] == support
    support["gate"]["delta"] = [[0.1]]
    with pytest.raises(ValueError, match="Only support"):
        fixed_supports(assets, arms, {"gate": 5}, rows=3, groups=2)


def test_schedule_requires_every_original_sample_in_original_order():
    history = [{"example_indices": [0, 2]}, {"example_indices": [1, 0]}]
    assets = {"parent-fits.json": [{"id": "fit", "history": history}]}
    assert verify_schedule(history, assets, "fit")["sample_indices_exact"]
    with pytest.raises(ValueError, match="sampled"):
        verify_schedule(list(reversed(history)), assets, "fit")
    with pytest.raises(ValueError, match="sampled"):
        verify_schedule(history[:1], assets, "fit")
