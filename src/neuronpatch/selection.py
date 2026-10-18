"""Selection baselines. These functions do not establish causal importance."""

import torch


def random_indices(width: int, count: int, *, seed: int = 0) -> list[int]:
    """Sample without replacement using a local CPU RNG, preserving global RNG state.

    Call once per target module with the guided selector's per-layer count to
    construct a layer-matched random baseline.
    """
    if type(width) is not int or type(count) is not int or not 0 < count <= width:
        raise ValueError("require integer dimensions with 0 < count <= width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(width, generator=generator)[:count].sort().values.tolist()
