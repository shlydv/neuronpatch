import json
from types import SimpleNamespace

import pytest

from experiments.kaggle_submit import push_bounded


def setup(
    tmp_path,
    *,
    owner="shlydv",
    private=True,
    account="shlydv",
    script="run.py",
    shape="NvidiaTeslaT4",
):
    calls = []
    api = SimpleNamespace(
        get_config_value=lambda key: account if key == "username" else None,
        kernels_push=lambda *a, **kw: calls.append((a, kw)) or "accepted-placeholder",
    )
    (tmp_path / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{owner}/neuronpatch-test",
                "is_private": private,
                "code_file": script,
                "machine_shape": shape,
            }
        )
    )
    (tmp_path / "run.py").write_text("raise RuntimeError('fixture must never execute')\n")
    return api, calls


def test_passes_runtime_to_sdk_without_changing_or_executing_the_bundle(tmp_path):
    api, calls = setup(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert push_bounded(api, tmp_path, timeout_seconds=5400) == "accepted-placeholder"
    assert calls == [((str(tmp_path.resolve()),), {"timeout": "5400"})]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "options,limit",
    [
        ({}, None),
        ({}, True),
        ({}, 0),
        ({}, 10801),
        ({"owner": "company-account"}, 5400),
        ({"account": "company-account"}, 5400),
        ({"private": False}, 5400),
        ({"shape": "UnreviewedAccelerator"}, 5400),
        ({"script": "../outside.py"}, 5400),
    ],
)
def test_rejects_invalid_or_unsafe_submission_before_calling_sdk(tmp_path, options, limit):
    api, calls = setup(tmp_path, **options)
    with pytest.raises(ValueError):
        push_bounded(api, tmp_path, timeout_seconds=limit)
    assert not calls
