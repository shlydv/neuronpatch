# Qualify the preservation rule on known successful and harmful patches

This is a bounded diagnostic, not an improved adaptation result. The
[previous protected-training pilot](../guarded_training/complete_v1/REPORT.md)
stalls while its conditional KL guard also rejects valid alias changes. The
saved-endpoint audit lacks probabilities for two genuine unrelated errors.
This replay fills that specific evidence gap before another training sweep.
**Complete:** the [report](complete_v1/REPORT.md) verifies840 prefix vectors and
96 generations. The frozen qualification fails: both known errors are caught,
but two older greedy-successful controls show substantial probability loss.
The [launch record](launch_v1.json) and source remain fixed.

## Fixed controls and prospective decision

Restore six already trained low-rate endpoints on the pinned FP16
Qwen3-4B-Instruct-2507 base: selected paired neurons, random seed11 and official
rank-one MLP LoRA, each on the same two exposed counterfactual edits. Four
LoRA/random endpoints are required positive controls. The case1 selected patch
is a required negative control because its saved answers replace the Lakers'
country with the injected target and give O₂ for oxygen's chemical symbol.
The case0 selected patch is an additional descriptive control. No new fitting,
supports, hyperparameters, fresh facts or targets are introduced.

The [protocol](protocol_v1.json) freezes16 probe rows: the six original auxiliary
rows per case, oxygen in both cases, and BBC/Lakers in case1. BBC includes the
known valid formal UK answer that the older scorer missed. Accepted finite
strings include the supplied aliases, formal UK variants and single final-period
variants. They are enumerated before collecting their missing probabilities.
This finite set does not capture every semantically valid string.

Questions where the base output is outside the supplied reference set are
reported as unresolved reference disagreements and excluded from retention
gates. This neither adjudicates their truth nor silently turns retention into
another supervised correction. There are14 eligible rows across the two cases.

The prospective per-question retention limit is an absolute0.02 drop in
accepted-response probability from the base. Qualification requires all four
positive controls to stay within that limit on every eligible row, retain
accepted complete answers and reproduce all requested transfers. Both known
negative rows must separately exceed the probability-drop limit and reject
the generated answer. A failure of any prerequisite blocks qualification;
the threshold will not be changed after observing the probabilities.

## What is measured

Each state directly evaluates every unique response prefix once, storing full
next-token vocabulary logits. This avoids imputing missing alias branches or
assuming that teacher forcing different longer strings gives bitwise-identical
prefix logits. Finite-response probability includes all tokens and EOS; duplicate
paths count once. Cached greedy generation is measured separately. The
independent verifier recomputes log-softmax and sequence sums from saved vectors.

All original correction/auxiliary/validation teacher records must replay exactly
at each merged endpoint, and every selected probe and transfer answer must match
its saved generated answer exactly. Base fingerprints and generation controls
are checked after each restoration. The full study has840 prefix forwards,
96 observed generations plus12 base-restoration generation controls. It creates
no optimizer. Inference uses CUDA0 of the free T4 allocation.

The worker has an1800-second internal model/replay limit and2100-second process
limit; setup subprocess limits are300/300/900seconds. Submission requests a
3600-second provider limit. These are runtime bounds, not a bound on queue wait,
and provider enforcement remains unverified until exercised.

Known positive/negative fixtures are calibration data. Even a successful result
would not establish prevention during fitting, protection of unseen tasks,
neuron-selection superiority, general safety or a recommended library recipe.
The full goal and next method hypothesis remain in the
[current decision](../../docs/current-method.md).

```sh
PYTHONPATH=.:src python -m pytest -q tests/test_outcome_guard.py tests/test_response_sets.py
```

The actual worker is tested on tiny FP32/FP16 Qwen models with real saved
LoRA/paired/random endpoints, complete replay, full missing-prefix collection,
independent verification and corruption rejection. Those checks validate the
machinery; they are not pretrained-model qualification results.
