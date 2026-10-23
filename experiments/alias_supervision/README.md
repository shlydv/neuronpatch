# Isolate auxiliary answer-equivalence supervision

**Completed:** [results and full-output review](complete_v1/REPORT.md).
The frozen primary prediction fails. Random and LoRA reduce the oxygen competing
continuation; selected neurons still answer O₂, although their Lakers answer is
repaired. All retain the requested edits; none meets the full preservation limit.
The design below preserves the prospective protocol rather than revising it.

The [completed replay](../outcome_guard/complete_v1/REPORT.md) shows latent
competition with ₂ after O even in greedy-successful LoRA/random endpoints.
Its logit-gradient audit also shows strong pressure to suppress EOS after an
already accepted United States answer because a canonical auxiliary label
continues with ` of`. Whether that pressure causes downstream errors remains
unproven. This bounded comparison changes that objective alone.

## Fixed comparison

The [protocol](protocol_v1.json) uses the same two exposed counterfactual facts,
three correction responses, six auxiliary examples, neuron supports, official
LoRA configuration, seeds, learning rate0.0003,50 updates, sample ordering,
loss scaling, clipping and Adam settings as the original comparison. Correction
loss remains unchanged. On the five initially accepted auxiliary rows per case,
sum the probabilities of the **existing reference aliases**, then divide the
negative log by the original canonical response length including EOS.

Do not average losses across aliases or normalize each alias by its own length.
Duplicate token paths count once. A singleton set calls the exact original
canonical CE implementation. Multi-alias prefix logits retain autograd while
offloaded to FP64 CPU; their model activations remain on the GPU. Shared
prediction prefixes use their first full-response teacher execution, with the
canonical response first, rather than assuming floating-point prefix invariance.
This numerical convention differs from the direct-prefix evaluation policy;
evaluation uses the same frozen direct-prefix policy across all endpoints.

The two legacy base/reference disagreements remain explicitly identified and
receive the same canonical CE in both arms. This preserves the intervention
contrast; it does not adjudicate those labels or recommend treating them as
ordinary retention data. Singleton eligible examples have the same loss exactly.
Oxygen and Lakers are not added to training. Neither prompts nor output formats
are tweaked. No new selector or hard protection controller is tested here.

## Eight fits with gating controls

First rerun canonical case1 selected neurons and LoRA. Their complete50-step
histories, exported weights, teacher logits, all generated banks and probe
vectors must exactly match the previously saved endpoints. Any mismatch stops
the comparison before accepted-set fitting. Other matched canonical references
come from the already verified original study; not all six trajectories are
retrained in this run.

Then train accepted-set supervision for selected paired neurons, random seed11
and official LoRA in both cases. Each new fit undergoes actual export, unload,
reload and merge. Reloaded parameters and canonical teacher logits must exactly
match their live values. Evaluate every original answer bank in the merged
model and collect the same finite-response probes used in the completed replay.
The run has544 generated observations and1084 saved full-vocabulary prefix
vectors. Independent verification checks decoding, marginal probabilities,
alias exposure, exact canonical controls and final artifacts. Full output
review is still necessary to separate valid wording from genuine errors.

## Prospective interpretation

The causal prediction requires at least two of three representations, including
selected neurons, to halve case1's P(₂ | oxygen prompt,O), retain4/4 requested
transfer answers in both cases, and preserve or restore the correct oxygen
answer. This would support the auxiliary-loss explanation on these exposed
development cases; it would not establish a robust adaptation method.

A practically useful candidate additionally needs all8/8 transfers, no new
genuine preservation errors or truncations, and the unchanged0.02 probability-
drop target on eligible probes. Report every probability and complete-answer
trade-off even when these criteria fail. An improvement shared by LoRA and
neuron patches is objective repair, not neuron-selection superiority. Later
claims still need fresh cases, budgets, seeds, another task family and total-cost
comparisons. The full [library/method goal](../../docs/current-method.md) remains.

## Resource scope and checks

This is one private free-T4 Kaggle job, with a requested7200-second provider
limit,5100-second worker-process limit and4800-second internal model/experiment
limit. Setup subprocess limits are300/300/900seconds. Queue wait is outside
these runtime scopes; provider enforcement is not assumed. Peak allocated CUDA
memory and fitting/export durations are recorded, including the extra alias
forwards. No whole-pipeline speed improvement is claimed from fewer parameters.
Read-only snapshots separate task and auxiliary gradient contributions before
clipping, so changes in loss alignment and clipping can be inspected alongside
the EOS hypothesis. Accumulation-rounding effects remain; these snapshots are
not independent auxiliary-only backpropagations or neuron-level attribution.
Both canonical controls include the observer and must still exactly replay.

```sh
PYTHONPATH=.:src python -m pytest -q tests/test_alias_loss.py tests/test_alias_worker.py
```

Checks compare full-model parameter gradients with an independent complete
probability graph in FP32/FP16, exact singleton CE, incompatible prompts, the
actual eight-fit tiny-Qwen worker and corrupted probability rejection. Separate
regression checks exercise the original paired worker after adding its optional
auxiliary-loss callback. These establish implementation behavior, not a
pretrained-model result.
