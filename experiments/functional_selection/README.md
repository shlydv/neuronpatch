# Proposed-update gain and functional risk

This experimental selector measures a different quantity from correction-gradient
magnitude: the predicted task benefit and conditional-output disturbance of an
explicit proposed parameter update. It remains outside the public API. CPU
oracles and tiny-Qwen checks pass; no pretrained outcome is established yet.

The frozen [protocol](protocol_v1.json) and [source identity](frozen_v1.json)
precede inference. The [first Kaggle submission](launch_v1.json) was accepted
with the exact frozen 150348-byte script and 37 bundled files. It was deliberately
cancelled to control resource use after the missing provider limit was found;
only an empty persisted log was available. This gives no scientific outcome.
A [single operational retry](retry_v2.json) is accepted as version 2 with
byte-identical source and a requested 5400-second provider session limit. It is
running at the recorded status check, and worker logs confirm dependency
installation. No pretrained calibration result is available yet. No setting or result
from the unavailable first run selects the method or changes the frozen criteria.

The [completed stopping study](../paired_checkpoints/complete_v1/REPORT.md)
found fast fitting but two unrelated errors already present at12 updates. It
also found that a later Adam step could oppose correction through historical
momentum. A useful next selector must consider both preservation and update
direction, with finite-step behavior checked before a longer fit.

## What is measured

For shortlisted neuron i, materialize only its incoming rows and matching
outgoing column from the factorized, equal-example mean correction gradient
g_i. The fresh, unclipped Adam proposal is

\[
u_i=-\frac{g_i}{|g_i|+\epsilon},\qquad
d_i=\frac{u_i}{\|u_i\|_2},\qquad
a_i=-g_i^T d_i.
\]

All coupled projections contribute to the same normalization. The gradient-only
direction is also implemented as a numerical control. Neither direction models
later Adam moments or global gradient clipping.

For a fixed normal-response prediction position, let p=softmax(z). With
independent Rademacher noise ξ, form

\[
r=\sqrt p\odot\xi-p\sum_v\sqrt{p_v}\xi_v.
\]

Its second moment is diag(p)-ppᵀ. Independent position noise and division by
√T give equal weight to responses of different lengths. Backpropagating the
detached coefficients through logits produces a VJP v whose squared dot with
the proposal estimates conditional Fisher curvature. This avoids the zero
gradient obtained from KL(reference||reference).

For each neuron, estimate R_i=mean[(vᵀd_i)²]. The ranking proxy is

\[
s_i=\frac{a_i^2}{R_i+\lambda\|d_i\|_2^2},\qquad
\lambda=\frac{\sum_i R_i}{\sum_i\|d_i\|_2^2}.
\]

This comes from optimizing one scalar step along a fixed direction under a
local quadratic penalty. Damping is computed on eligible candidates, not tuned
against evaluation outcomes. All-zero measured risk is explicitly flagged;
it is not evidence of preservation.

The score treats neurons separately. For a selected set S, the joint prediction
instead sums probe dots before squaring:

\[
\widehat{\mathrm{KL}}(\alpha,S)
=\frac{\alpha^2}{2M}\sum_{m=1}^{M}
\left(\sum_{i\in S}v_m^T d_i\right)^2.
\]

This retains between-neuron cross terms, including cancellation or amplification.
It can disagree with summing individual risks. The finite-step calibration
records both estimates and actual full-vocabulary anchored KL.

## Checks

The exact softmax KL Hessian agrees with the probe second moment on a complete
orthogonal-noise oracle. Separate dense weight gradients agree with compact
incoming/outgoing VJPs and with proposed first-step PyTorch Adam directions.
Tests also cover cross terms, caller RNG, failure cleanup, FP32/FP16 tiny Qwen,
and exact finite-step restoration on exceptions. These check mathematics and
execution, not semantic selection quality.

~~~sh
PYTHONPATH=.:src python -m pytest -q tests/test_functional_selection.py
~~~

## Bounded pretrained calibration

The original submission sets an internal 3600-second worker deadline and a
3900-second worker-process timeout. It does not set a provider session timeout
or bound dependency setup. These are different scopes; the original job must
not be described as having a bounded total session runtime.
The [submission-limit audit](submission_limits_v1.json) records this gap.
Future submissions use `experiments.kaggle_submit` with a required
`--timeout-seconds` value and record it in launch provenance. The provider
runtime limit does not establish a queue-wait deadline. This change does not
resubmit, cancel or modify the existing calibration.

The prepared protocol uses the two existing development facts and the same
per-layer coordinate budget. The algorithmic shortlist contains the top64
original gradient-ranked neurons per layer. Random seed11 indices are added
for evaluation but excluded from algorithmic selection and damping.

Compare original gradient magnitude, proposed Adam gain, gain/risk and random
supports, using the same proposed directions. Apply scales0.01 and0.03 and
restore the exact base after each:16 data-informed one-step edits total.
Only the original six preservation-training response contexts determine risk.
Exposed scope/utility questions are held out of selection. Both scales get
teacher measurements; scale0.03 also gets free-running answers. No scale is
selected after observing results, and no multi-step optimizer is run.

At scale0.03, gain/risk must lower held-out mean KL by at least20% versus gain-only
in both cases while retaining at least half its positive measured correction
improvement, and itself improve correction NLL by at least0.001. It must
also yield a25% improvement in correction-gain squared per unit held-out KL;
the gain-only baseline must have held-out KL at least1e-8 to make this informative.
Scale0.01 checks predicted versus observed joint calibration KL, with ratio0.25–4 required
where observed KL is at least1e-8 and at least one informative condition per case.
All outputs, discovery costs and precision
differences are retained even when these criteria fail.

The separate `diagnose` module is a post-hoc check of sketch stability. It
reselects supports while holding out an entire preservation context or one
independent probe per context, then measures their held-out sketch response.
It also reports joint versus diagonal curvature to expose cross terms. These
fold supports are never applied to the model and do not alter the frozen
interventions or success criteria. The check can expose sensitivity to noise
or the small anchor bank; stable selection alone does not establish preservation.

~~~sh
PYTHONPATH=.:src python -m experiments.functional_selection.diagnose \
  --results PATH_TO_VERIFIED_RESULTS --integrity PATH_TO_INTEGRITY_JSON \
  --output PATH_TO_NEW_DIAGNOSIS_JSON
~~~

Success would justify a trained-method comparison with strong LoRA controls;
it would not demonstrate an improved library recipe. Fresh cases, lower budgets,
another task family and measured retention remain necessary.

## Limits and provenance

These are teacher-context conditional-token distributions including EOS, not an
expectation over all reference-generated sequences. The local quadratic model
can fail under finite changes and FP16 rounding. Two probes per anchor and a
small normal bank can under-estimate risk or overfit candidate selection.
Shortlisting can exclude useful lower-gradient neurons. A score at the base
does not constrain later optimization. No unique concept/circuit label follows.

The [cached-context audit](context_audit_v1.json) quantifies one part of this
scope. Only3/6 and4/6 canonical anchor responses exactly equal the corresponding
greedy base token sequences. In each case,13/19 canonical prediction contexts
occur along those greedy responses. Because the actual metric weights anchors
equally, the corresponding shared-context fractions are75.56% and80.00%.
A next-token disagreement still counts as a shared prediction context when
its preceding tokens agree. Different valid aliases can cause divergence;
these counts are not error rates or sequence probability mass. They establish
neither the cause of interference nor that another context choice performs better.
The audit uses no model calls and independently checks900 short prefix pairs:

~~~sh
python -m experiments.functional_selection.context_audit \
  --parent PATH_TO_ORIGINAL_RESULTS --integrity PATH_TO_ORIGINAL_INTEGRITY \
  --output PATH_TO_NEW_CONTEXT_AUDIT
~~~

[DCO](https://arxiv.org/html/2609.13680v1) formalizes adaptation under an anchored
behavioral-drift budget and relates local drift to Fisher geometry. Its inspected
[official source](https://github.com/CONE-MT/DCO/tree/222f26c51c4ca8efa06c5d077c3b9c82d55c018f)
uses layer-freezing training configurations; this is not its reproduction.
The inspected KL-analysis script measures KL(edited||base), whereas our proposed
metric explicitly uses KL(base||edited). Neither its training results nor this
new selector's quality have been reproduced here.
[Earlier subset-selection work](https://aclanthology.org/2024.findings-eacl.122.pdf)
already combines target and pretraining gradient statistics. Functional-risk
selection and constrained optimization have prior art; this code does not make
a novelty or superiority claim. No upstream implementation is incorporated.
