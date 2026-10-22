# Fingerprint allocation benchmark

The adapter checks a strict base-model fingerprint before attaching or merging.
Hashing now reads the contiguous byte buffer directly, rather than creating an
additional Python byte string the size of each parameter. It still reads every
byte and retains the same shape, dtype and parameter-name headers. It does not
remove CPU transfers or copies needed to make noncontiguous tensors contiguous.

Run from the repository with the package installed:

```sh
python -m benchmarks.fingerprint --output /tmp/neuronpatch-fingerprint.json
```

The [recorded run](fingerprint_cpu_v1.json) uses a synthetic 64 MiB CPU tensor
and seven interleaved timing repetitions. Both versions return the same digest.
Additional Python allocation during hashing falls from 67,109,433 bytes to 976
bytes. Median measured times are 21.92 ms and 20.88 ms respectively. These are
one-machine microbenchmark observations, not replicated end-to-end performance
claims. Allocation is measured separately from timing with `tracemalloc`; it is
not process RSS or GPU memory.

The legacy-identity test also checks BF16, noncontiguous tensors, scalar and
empty tensors, booleans and signed-zero bits against a digest recorded from the
unmodified implementation. Existing artifact identities remain unchanged.

This is a memory-efficiency improvement, not a model-quality result. The
already-frozen Kaggle checkpoint study uses its original implementation.
