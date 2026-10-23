# Compare learning methods, not only individual repairs

23 October 2026. The full library goal is unchanged and unachieved. This note
records what the source review changes about experimental design; it does not
announce a new benchmark run or a working method.

## A mismatch we should stop carrying forward

A small *selection/calibration* dataset, a small number of *trainable parameters*,
and a small *training* dataset are different claims. Our small counterfactual
edits did not reproduce broad behavior-learning experiments simply because both
updated few coordinates. The audited NeST setup uses many labeled training
examples. Its shared-row mechanism may be relevant, but its headline does not
predict success from three phrases for one changed fact.

[LoRA-One, sections 6.2–6.3](https://arxiv.org/html/2502.01235v3) uses 100K
MetaMathQA examples in its smaller math comparison, rank eight, method-specific
hyperparameter tuning and five runs. Its reported direct-prompt GSM8K gain over
ordinary LoRA is about 1.18 percentage points. The larger math experiment uses
395K examples for four epochs. Its smaller-data/one-step GLUE results concern a
different model/task setting. We must not conflate those protocols or expect a
few development questions to resolve a similarly sized advantage. These are
published results, not our reproductions.

The completed relation probe qualifies a repeated development weakness: 75/96
correct, with 24/24 target-only controls intact. This warrants a learning pilot
but establishes no selected-neuron superiority. Proceed with that concrete lead;
do not launch math training in parallel merely because it is another option.

## Selection needs stronger, correctly named baselines

[FISH](https://arxiv.org/html/2111.09839) already constructs a fixed sparse mask
from large diagonal Fisher estimates, with a supervised empirical variant using
per-example squared gradients. This is precedent for data-dependent sparse
fine-tuning, not neuron interpretation. It differs from taking the squared mean
gradient. A whole-neuron aggregation would be our structured variant, not an
exact FISH reproduction. The paper distinguishes the selection-sample budget
from subsequent optimization and distinguishes true from empirical Fisher.
We inspected its method section and official repository documentation, not a
full implementation or replicated training result.

For gradients g_i of a given group, averaging squared norms retains cross-example
variation that the norm of the average removes:

    mean_i ||g_i||^2 = ||mean_i g_i||^2 + mean_i ||g_i - mean_i g_i||^2.

This algebra does not prove which score will select better neurons. In particular,
variation can reflect different legitimate targets as well as noise. Our existing
cross-prompt agreement code estimates worst first-order gains along a mean
direction for paraphrases of the same target; it is not an empirical-Fisher mask
or evidence of transfer across different examples. Before inventing another
selector, compare these existing alternatives on a common learning task.

The [coordinate audit](selection-coordinate-audit.md) adds a narrower warning:
reciprocal up-row/down-column rescaling preserves a gated MLP's function while
changing raw group-gradient rankings. This is verified in a tiny algebra check,
not on the language-model task. Gate-based sensitivity has established pruning
precedent and does not by itself predict adaptation quality. The current pilot
remains frozen; do not replace its selector in response to this observation.

## A concrete candidate for the next comparison

A fixed skill-learning benchmark would evaluate adaptation even when there is no
single memorable factual error. [GSM8K](https://github.com/openai/grade-school-math)
is one candidate: it provides training/test separation and worked arithmetic
solutions. It is old and may have appeared in pretraining, so using it would not
establish unseen knowledge or originality. We have not downloaded its training
set or launched math fine-tuning in this phase.

For such a comparison, freeze a training pool, development split and untouched
confirmation split *before* inspecting model errors. Include a reasonable
reasoning prompt baseline; a terse arithmetic failure is not enough qualification.
First establish that ordinary LoRA learns on that fixed task with complete-response
supervision and a bounded tuning budget. Then compare matched-cost random sparse,
existing task-gradient selection and empirical-Fisher/group scoring, with exactly
the same training and development opportunities. Gradient-informed low-rank
initialization is a relevant stronger baseline if discovery is our claimed gain.

Report uncertainty and total discovery/training/evaluation costs. A low-data pilot
would be a pilot, not a reproduction of a 100K-example paper or proof of a one-point
advantage. Held-out transfer, unrelated utility, multiple seeds and another task
remain necessary for the requested robust-library claim. No new selector or
router should be built merely because it is easy to implement.
