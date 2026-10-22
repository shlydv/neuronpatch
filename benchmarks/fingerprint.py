"""CPU hashing microbenchmark;not a GPU training or model-quality benchmark."""

import argparse
import datetime
import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import torch

from neuronpatch.patch import _fingerprint_state


def legacy(state):
    """Frozen c80dfbf baseline,retained only for a reproducible comparison."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def measure(mib, repeats):
    assert mib > 0 and repeats > 0
    torch.set_num_threads(2)
    state = {"weight": torch.arange(mib * 1024 * 1024 // 4, dtype=torch.float32)}
    expected = legacy(state)
    assert _fingerprint_state(state) == expected
    functions = {"legacy_bytes_copy": legacy, "contiguous_buffer": _fingerprint_state}
    samples = {key: [] for key in functions}
    for i in range(repeats):
        order = list(functions) if i % 2 else list(reversed(functions))
        for key in order:
            started = time.perf_counter()
            value = functions[key](state)
            samples[key].append(time.perf_counter() - started)
            assert value == expected
    peak = {}
    for key, function in functions.items():
        tracemalloc.start()
        function(state)
        _, peak[key] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": (
            f"CPU synthetic {mib} MiB hashing microbenchmark; "
            "peak is additional Python allocation, not process RSS. "
            "No LLM weights or GPU execution. Does not measure training or full export latency."
        ),
        "tensor_bytes": mib * 1024 * 1024,
        "repeats": repeats,
        "fingerprints_identical": True,
        "fingerprint": expected,
        "torch": torch.__version__,
        "median_seconds": {k: statistics.median(v) for k, v in samples.items()},
        "peak_python_allocation_bytes": peak,
        "samples_seconds": samples,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mib", type=int, default=64)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = measure(args.mib, args.repeats)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples_seconds"}, indent=2))


if __name__ == "__main__":
    main()
