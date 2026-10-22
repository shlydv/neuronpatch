# Risk-aware selection helps the local tradeoff, but the risk estimate fails

**Complete, independently recomputed, and reviewed. The frozen advancement gate fails.**

This is the unchanged functional-direction experiment, executed as Kaggle
provider version 2 after the acknowledged resource-control cancellation of
version 1. It is not a second scientific trial. On two exposed development
facts, selecting for gain and conditional Fisher risk reduces held-out output
drift while retaining almost all the immediate correction-loss improvement.
However, the local risk estimate is too optimistic in case 0. None of these
single updates produces the requested complete edited answer. This is useful
selection evidence, not a trained method or an advantage over LoRA.

## What was tested

The [frozen protocol](../protocol_v1.json) applies sixteen restored one-step
weight changes to pinned Qwen3-4B-Instruct-2507. Four supports—gradient magnitude,
fresh-Adam gain, gain/risk, and random seed 11—use the same direction family,
173 paired neurons and 1,328,640 scalar coordinates. Both predeclared scales,
0.01 and 0.03, receive full-vocabulary teacher measurements. Only 0.03 receives
free-running answers. No multi-step optimizer fit or LoRA comparison occurs here.
Only six original preservation-training contexts influence risk selection.
Scope/utility inputs are excluded from selection, but remain exposed development
data rather than fresh confirmation.

## Positive result and failed criterion

At the primary scale 0.03:

| Case | Gain-only correction NLL improvement | Gain/risk improvement | Gain-only held-out KL | Gain/risk held-out KL | Drift reduction | Gain²/KL efficiency ratio |
|---|---:|---:|---:|---:|---:|---:|
| 0 |5.71721|5.59432|0.349784|0.200674|42.63%|1.669|
| 1 |5.58419|5.59052|0.701102|0.264982|62.20%|2.652|

Both predeclared improvement/disturbance comparisons pass. NLL improvement is
an average teacher-forced training loss reduction, not answer accuracy.
Random changes disturb outputs much less, but make only 0.959 and 0.601 NLL
improvement; these results do not support dismissing random after a longer fit.

The smaller-scale prediction check fails. In case 0, predicted/observed
calibration KL is 0.069 for gradient magnitude, 0.074 for gain-only, and 0.203
for gain/risk. All fall below the frozen 0.25 threshold. Random passes at 0.538.
All four conditions pass in case 1, with ratios 0.743–1.490. Therefore the
overall frozen gate remains failed; the improvement criterion is not substituted
for the calibration criterion. See [analysis](analysis.json).

## Full answers limit the interpretation

All four methods achieve **0/8 requested counterfactual transfer answers** after
their one-step changes. The guided case-0 answers move toward Italian or
Italian-American rather than the injected target. Some case-1 answers become
long explanations or qualifications instead of the requested target. Four
generated observations hit the 32-token limit; no unexpected special token occurs.

Frozen preservation-score losses total four for gradient/gain-only, three for
gain/risk, and zero for random, out of 49 initially accepted observations.
Reading all 62 distinct prompt/reference/output contexts, covering 310 generated
observations, identifies these losses as country-name/demonym changes such as
United States → American and United Kingdom → British, rather than new clear
factual corruption. The Lakers answer becomes America under the gain-only
conditions and remains United States under gain/risk. Both express the same
country here. The pre-existing England/UK granularity issue, Unicode water
formula, and full triangle completion remain legacy scorer limitations.

This review is by the assistant, not a blinded human. No new clear preservation
error identified on this small bank does not establish general preservation.
In particular, none of the methods has completed the desired edit. The
[review coverage](review.json) keeps these limits separate from frozen scores.

## Why the risk prediction was optimistic

A [post-hoc saved-logit audit](logit_path.json), with no new model calls, evaluates
the exact softmax KL and its base quadratic form on the *observed* logit change.
This is a finite logit secant, not an exact measured parameter Jacobian.

For case 0 at scale 0.01:

| Support | Original sketch prediction | Quadratic form of observed logit change | Exact observed KL |
|---|---:|---:|---:|
| Gain-only |0.00030167|0.00026864|0.00408341|
| Gain/risk |0.00005307|0.00006821|0.00026179|

The sketch is reasonably close to this observed-displacement quadratic form,
while the exact KL is 15.2 and 3.8 times that form. Thus the evidence does not
support treating additional random probes alone as the remedy. The quadratic
softmax approximation itself misses much of the finite response change.

As a diagnostic, extrapolate the saved 0.01 logit change by three and evaluate
the exact softmax KL, rather than scaling a quadratic estimate by nine. For
gain/risk this predicts calibration KL 0.32634 versus observed 0.35830 in case 0,
and 0.02523 versus 0.02701 in case 1. These are retrospective calculations on
the same contexts; they are not a prospectively validated step controller.
Network path changes and FP16 rounding still contribute and have not been
separated from one another.

The [sketch audit](sketch_diagnosis.json) also finds coherent cross-neuron
effects: joint/diagonal curvature ratios are 9.18 and 8.63 for gain/risk, versus
26.39 and 21.10 for gain-only. Diagonal risks would substantially underestimate
joint disturbance. Independent-noise split selections share 149 and 141 of
173 neurons, respectively. Held-out noise responses favor the risk-aware
selection, but some held-out anchors do not. Stability on six contexts is not
evidence of safety on new questions.

## Consequence for the method

Keep the useful ranking signal experimental. Do not use its base quadratic
value as a certificate for a finite update, or advance the unchanged predictor
into a longer fit. The next candidate must measure the actual preservation
change after a proposed update, reject/backtrack excessive changes, and restore
optimizer state as well as parameters on rejection. Apply the same protection
to random selection and LoRA. Completing edits, preservation on unseen inputs,
and total adaptation cost remain the relevant outcomes.

This changes the role of the risk score from a proposed safety estimate to an
initial support-ranking heuristic. Measured acceptance, anchored KL and line
search have prior art; their presence alone is not our contribution. A new
trained comparison must earn an advantage before fresh confirmation or a
recommended library recipe.

## Execution and verification

The [Kaggle run](https://www.kaggle.com/code/shlydv/neuronpatch-functional-selection-v1-20261022)
returns 0 after 460.95 seconds (7.68 minutes), including dependency installation
and model loading. It requested a 5400-second provider runtime limit, and exited
normally; the provider's timeout enforcement was not exercised. The earlier
unavailable version-1 execution consumed account GPU quota before cancellation
and must remain part of research cost reporting; 7.68 minutes is this successful
retry's runtime, not the total cost of the experiment history.

All 37 accepted bundled file hashes, the execution submission and the protocol
match the frozen source. The independent verifier recomputes all 16 supports
and update arithmetic, 18 teacher records, and all 310 decoded observations.
The worker confirms exact base restoration and 32 extra baseline generation
controls. See [source integrity](source_integrity.json), [integrity](integrity.json),
[completion](completion.json), and [provider](provider.json). Three separate
logit-path tests cover an independent KL oracle, shift invariance, severe
softmax curvature on a linear logit path, curved paths and equal-example weighting.
