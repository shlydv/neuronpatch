# Joint sparse updates: a candidate, not an established method

The intended result remains better useful adaptation than strong baselines.
The [outcome-guard replay](../experiments/outcome_guard/complete_v1/REPORT.md)
failed its strict qualification: some greedy-successful controls have substantial
latent probability interference. The next step is a matched auxiliary-objective
ablation, not a claim of a qualified protected method. This note
separates three experimental choices that otherwise become an uninterpretable
bundle: the outcome objective, the trainable representation and the controller
that turns gradients into accepted updates. No model result is claimed here.

## Relevant prior art and what it rules out claiming

[GEM](https://arxiv.org/html/1706.08840) projects a proposed gradient to the nearest
one satisfying constraints derived from previous-task gradients. Its dual
quadratic program is small in the number of constraints, even for many network
parameters. The [official implementation](https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py)
uses a regularized dual solve and SGD. Thus coordinating an update through
gradient constraints is established work. The repository's stated noncommercial
license also means its code should not be copied into this MIT project; the
mathematical idea can be discussed and independently implemented with attribution.

[A-GEM](https://arxiv.org/html/1812.00420) replaces those multiple constraints
with a single gradient of an average replay loss. This reduces computation, but
changes the protection objective. For our requirement, one damaged question
cannot be hidden by improvements on other questions. Its efficiency/average
trade-off is relevant; the average constraint alone is not our desired guard.

[GPM](https://arxiv.org/html/2103.09762) obtains layerwise bases from activations
using SVD and projects updates away from those subspaces. It supplies another
precedent for protecting representations, rather than training only selected
neurons. Applying an independent sparsity mask after such a projection need not
preserve its orthogonality. Our earlier
[restricted-projection note](../experiments/fine_reference/SPARSE_PROJECTION_NOTE.md)
already derives that compatibility issue. A dense-projection result cannot be
transferred to a sparse update without checking the actual restricted operator.

These works do not establish that our proposed combination improves an LLM.
Nor does a local gradient condition certify a finite optimizer step or unseen
questions. They identify strong controls and prevent a false novelty claim.

## One concrete representation to investigate after the objective ablation

For selected neuron/group j, let D_j be a normalized, fixed proposal on its
paired incoming rows and outgoing column. Distinct neuron groups occupy
disjoint parameter coordinates. Their parameter-space inner products therefore
vanish, although their effects on model answers can reinforce or cancel.
Represent an update as

    theta(c) = theta_base + sum_j c_j D_j.

Coefficients may have either sign. Optimizing them jointly permits a combination
to preserve an answer even when separate per-neuron risk scores look large.
The hypothesis is that useful coordinated directions can be learned with fewer
optimizer variables and fewer correction examples. It may fail because the
fixed task-gradient directions are too restrictive or do not transfer between
question forms. Fixed-basis adaptation itself already has prior art.

Do not advertise173 coefficients as a173-number deployable patch: the present
directions would still contain about1.33million stored coordinates, roughly
5.3MB in FP32. The frozen base, activations, discovery and acceptance forwards
still dominate much of the actual memory/runtime. Measure peak memory, complete
wall time and saved bytes, with equally precise LoRA controls.

## A controller must protect the proposed step, not an intermediate gradient

Let p_i(c) be the probability of a supplied accepted-response event on retention
question i. The current qualification asks whether preserving this event is a
better contract than preserving every conditional vocabulary distribution.
With a qualified contract and remaining probability budget, a possible local
proposal is

    minimize_d  0.5 (d - v)^T M (d - v)
    subject to  J_i d >= p_i(0) - delta_i - p_i(c), for each retained i,

where v is the proposed coefficient displacement, M is an explicit positive
metric and J_i is the derivative of p_i at the current coefficients. This is
ordinary constrained optimization, not a new mathematical contribution.

Projecting a raw gradient and then applying Adam need not preserve the same
constraints. Either use a controller with a clearly defined update geometry,
or project its actual displacement. Optimizer-state handling must be specified
for accepted projected steps and fully rolled back on rejection. Re-evaluate
the nonlinear probability constraints after the actual finite parameter update;
the local approximation can be wrong. A failed solve or exhausted backtracking
must return a failed adaptation, not silently relax protection.

Even an exact accepted-event constraint cannot alone certify greedy generation.
For example, valid first answers A/B can change from probabilities0.58/0.01 to
0.295/0.295 while an invalid answer C stays at0.40. Accepted mass is unchanged,
but greedy decoding changes from A to C. An independent probability-tree check
in [the tests](../tests/test_response_sets.py) demonstrates this case with EOS.
The training controller must retain complete-answer checks on its calibration
questions as well as the differentiable event statistic; final fresh-answer
evaluation is still separate. This limitation is mathematical, not a claim that
the current full-model replay exhibits that particular failure.

At a highly confident base, accepted-event gradients may be tiny. At the exact
base, a base-anchored KL gradient is zero. Neither fact licenses pretending that
the model is insensitive to larger changes. The finite-check cost is real, and
overly conservative projection may simply stop all methods again.

## What an informative comparison must separate

First give ordinary LoRA and random supports the corrected outcome contract.
If they alone improve, the result is objective repair, not selected-neuron
superiority. Then compare selected versus random supports under both unrestricted
paired deltas and the coefficient representation, keeping the controller and
data identical. This separates selection from the restrictions imposed by the
representation. A gradient-informed low-rank control is also necessary when
claiming that task-gradient discovery reduces learning effort; count its
nonzero initialization as part of adaptation.

The present Lakers/oxygen examples are known calibration failures. Protecting
those exact examples during fitting does not demonstrate preservation on new
examples. Keep new retention inputs and transfer wordings out of selection and
stopping. Validation used to choose a checkpoint is development data. Count all
aliases, examples, probes and tuning opportunities in comparisons. Do not consume
the untouched confirmation facts until the complete recipe is frozen.

A useful result must improve a held-out correction/preservation/cost trade-off
against strong baselines across budgets, seeds and a second task family. A
smaller coefficient count, lower teacher loss, more confident training answers,
or a successful known-control classification is insufficient. This candidate
should be abandoned or revised if its complete behavior and total costs fail
that comparison; the library goal should not be redefined around it.
