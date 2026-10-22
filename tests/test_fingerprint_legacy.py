import torch

from neuronpatch.patch import _fingerprint_state


def test_buffer_hash_preserves_legacy_artifact_identity_including_bit_patterns():
    state = {
        "weight": torch.tensor([[0.0, 1.0, -2.0], [3.0, 0.5, -0.25]], dtype=torch.float32).T,
        "bf16": torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16),
        "signed_zero": torch.tensor([-0.0, 0.0], dtype=torch.float16),
        "count": torch.tensor(7, dtype=torch.int64),
        "empty": torch.empty((0, 2), dtype=torch.float64),
        "flags": torch.tensor([True, False]),
    }
    # Recorded from c80dfbf's unmodified byte-copy implementation, before this change.
    legacy = "139e8f131f6fcbf68001dea84e1ab0421f6d025fb85e5b328458b14489e36605"
    assert not state["weight"].is_contiguous()
    assert _fingerprint_state(state) == legacy
    assert _fingerprint_state(dict(reversed(list(state.items())))) == legacy
    state["signed_zero"][0] = 0.0
    assert _fingerprint_state(state) != legacy
