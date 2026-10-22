# Frozen endpoint replay

**Status: complete and independently verified.** The
[report](complete_v1/REPORT.md) finds all248 live/merged answer pairs exactly
identical, including the observed failures. All496 scored outputs and16 teacher
records pass verification;71 distinct contexts are reviewed. See the
[launch record](launch_v1.json); earlier [running notes](running_notes_v1.json)
are explicitly partial historical observations.

This diagnostic separates the behavior of a trained live patch from the behavior
of its FP16 merged deployment. It creates **no optimizer, no new fit, no new
question, no changed target and no new selector**. It uses eight saved endpoints
from the [completed comparison](../paired_mlp/complete_v1/REPORT.md): low/high-rate
gradient-paired, low-rate random seed11, and low-rate official LoRA for both facts.

The immediate unresolved observation is a high-rate paired fit whose last
sampled pre-update loss was tiny but whose final merged answers were broken.
Those measurements differ in both model state and prompt scope. We cannot
attribute the gap to merging or learning without replaying the saved endpoint.

For each endpoint, reload the exact FP32 patch on the pinned FP16 base, measure
all complete train/preserve/target-validation teacher tokens, and generate the
existing target/scope/utility test banks. Repeat after the original merge. The
496 scored observations cover both states. Two base generation controls also
run after each restoration. Require merged logits and generations to reproduce
the parent exactly; otherwise stop causal attribution and report the mismatch.

The [protocol](protocol_v1.json) binds all needed parent artifact hashes before
inference. Eight post-hoc development endpoints deliberately include a known
failure. This is a diagnosis of our implementation and saved behavior, not new
confirmation or evidence that a selector beats the baselines.

Actual tiny-Qwen worker tests exercise all eight replays in FP32 and FP16,
including official LoRA, restoration, exact parent replay, independent loss/KL
verification and rejection of an altered reported loss. Numerical checks do
not predict the real-model outcome. GPU replay is bounded to2700 worker seconds,
with a3000-second outer command timeout; dependency setup is additional.

Before interpreting this as a selection problem, resolve this deployment
uncertainty. Separately, the completed pilot already shows that almost-zero
preservation-training loss does not guarantee unrelated-answer preservation.
For case1, selected paired preservation CE is about4.48e-6 while clean random
seed11's is1.76e-5. Smaller training loss did not mean less collateral damage.
The current validation scope bank has only1/6 initially accepted answers, so
its preservation gate is weak. Both findings argue for representative functional
preservation evidence rather than assuming that stronger fitting solves safety.

```bash
python -m experiments.paired_endpoint.prepare --parent PARENT_RESULTS \
  --integrity PARENT_INTEGRITY.json --output DATA
python -m experiments.paired_endpoint.build --data DATA --output BUNDLE
python -m experiments.paired_endpoint.verify --parent PARENT_RESULTS \
  --results RESULTS --protocol experiments/paired_endpoint/protocol_v1.json \
  --output INTEGRITY.json
```

Do not tune or select a new method from the test outputs in this diagnostic.
Fresh confirmation inputs remain unused. A useful-library advantage still needs
new evidence against matched random and LoRA baselines.
