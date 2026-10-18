# Pretrained grounded-QA pilot: the selector failed the preservation gate

Run on 2026-10-18 using pretrained Qwen2.5-0.5B-Instruct on a local Apple GPU.
The [protocol](PROTOCOL.md) was written before results. This is synthetic fictional
record extraction with missing fields; it is not a natural-document QA benchmark.

| Method | Correct supported answers / 24 | Correct abstentions / 24 | Combined / 48 | Arithmetic / 16 |
|---|---:|---:|---:|---:|
| Prompted base | 24 | 13 | 37 | 16 |
| Activation-selected columns | 17 | 24 | 41 | 16 |
| Random columns, seed 11 | 24 | 24 | 48 | 16 |
| Random columns, seed 29 | 21 | 24 | 45 | 16 |
| Random columns, seed 47 | 22 | 24 | 46 | 16 |
| LoRA, same layers | 22 | 24 | 46 | 16 |

The selected patch fixed every missing-evidence error but unnecessarily abstained
on seven supported questions. Its supported-answer accuracy fell 29.2 percentage
points, exceeding the preset 5-point tolerance. All seven new errors were literal
`UNKNOWN` outputs, not scoring disagreements. The selected method's balanced
accuracy was 85.4%, versus 96.5% averaged across the three random controls.
LoRA's 8.3-point supported-answer drop also failed the strict preservation gate.
Only one random seed passed; it must not be presented as a reliably successful
method on the strength of that best seed.

## What this establishes

- A real pretrained model can be trained through neuronpatch, saved, and reloaded.
- This particular activation-contrast selector did worse than all three random
  selections and LoRA on the held-out task families. We have no evidence that it
  discovers especially useful neurons for this edit.
- Better aggregate accuracy can hide a bad trade-off: the selected patch gained
  four net correct answers while breaking seven previously correct answers.
- All methods passed the 16-item arithmetic sentinel. This tiny check does not
  establish preservation of general capabilities.

The effect of selection is confounded with the choice of a simple unnormalized
activation ranking, fixed layers, and a small training budget. This result does
not refute neuron-targeted fine-tuning in general, nor the cited papers' methods.

## Configuration and checks

Selected and random patches each trained 43,008 scalar parameters (16 columns in
three 896-output projections). LoRA trained 34,560 parameters on the same three
projections. Selected and LoRA each received two learning-rate candidates; all
random controls used the selected method's validation-chosen rate. All evaluated
fits used 0.001, identical training order and 32 steps. See raw validation results
for rejected candidates, including the unstable selected 0.003 fit.

Fresh test entities, field names and document families were not used for
selection, training, or hyperparameter choice. Calibration used the final prompt
token only, not gold answers. Every fictional record supplied one supported and
one missing-field question. Exact-match normalization removes punctuation, spaces
and case; the raw answers are published for audit.

The evaluated column patches were reloaded onto the exact base weights before
final testing. All four fixed-input reload checks had zero maximum logit error.
A separate pretrained selected-patch merge check is in `results/deployment.json`.
The full local test suite passed (20 tests, including the Apple GPU collector test).

Model revision, library versions, source/data hashes, selected indices, trainable
counts, per-fit training times, losses, all raw outputs, and metrics are under
[results](results/summary.json). Model weights and trained binary patches are not
committed; the selected patch is retained locally. Training times exclude loading,
calibration, fingerprinting, validation and generation, so they are not end-to-end
performance claims.

## What to do next

Do not add more features or claim a better fine-tuning algorithm. The next justified
research question is whether selecting neurons using both the correction objective
and the behavior-to-preserve objective reduces over-abstention. Compare that with
random selection and LoRA using the same tuning budget on a new preregistered test
set. The current final test is now development evidence and must not be reused as
an untouched benchmark. First inspect why these seven supported examples changed;
any resulting selector or training changes need fresh evaluation.
