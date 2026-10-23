# Current method decision — 23 October 2026

The goal is unchanged: a useful, robust sparse-adaptation library with a
verified advantage over strong baselines. We have working compact patches,
complete-response training and deployment checks. We do **not** have that
advantage. The auxiliary-supervision ablation is complete: it helps some outcomes
but does not fix the selected-neuron oxygen error. The [previous decision](decisions/2026-10-22-method.md) and
[experiment history](experiment-history.md) preserve earlier reasoning/results.
The [completed auxiliary-objective comparison](../experiments/alias_supervision/complete_v1/REPORT.md)
verifies eight fits, 544 generations and 1,084 prefix vectors. Its frozen primary
prediction fails: selected neurons retain the oxygen error, while random and LoRA
reduce its competing continuation probability. Selected neurons do repair Lakers;
LoRA loses accepted probability there. None meets the full preservation criterion.
The [additional primary-source review](research-2026-10-23.md) identifies close
curvature and component-coefficient precedents that constrain future claims.

The [retrieved-evidence baseline](../experiments/retrieval_evidence/complete_v1/REPORT.md)
is complete: all 110 outputs reviewed and independently checked. Both prompt
variants correctly abstain on 21/22 missing-evidence questions. Only one repeated
event-relation error survives, below the frozen four-question qualification.
Eleven responses hit the generation cap; these are recorded as incomplete, not
silently repaired or counted as complete successes. **Do not train on this slice.**
The explicit-prompt city-only venue answer is defensible and not a clear error.

The [precision/prompt diagnostic](../experiments/inference_sanity/complete_v1/REPORT.md)
also completed: 36 generations with all twelve token sequences identical across
single-device FP16, split-device FP16 and split-device FP32. All eight prior
inputs replay exactly. Two user-only prompt variants do not fix the factual
errors. These precision/device choices do not explain the observed failures;
no broader inference-correctness or model-quality conclusion is justified.
Do not run another generic dtype/prompt sweep on these cases.

The current decision is to close both qualifications without adaptation. One
event-relation confusion remains a possible failure-family lead, not a new
method or adequate training set. Further work must establish repeated, independently
varied errors and a common-supervision LoRA baseline. The
[scope assessment](scope-and-adaptation.md) separates conditional deployment from
sparse capacity and documents prior art; it is not a plan to build another router
without measured value.

The [counterbalanced relation probe](../experiments/relation_contrast/README.md)
is now submitted as Kaggle version 1. It uses four explicit event distinctions,
three fictional name sets, swapped candidate identities, reversed passage order,
target-only and missing-evidence controls: 96 correlated prompts, zero training.
It tests whether the observed relation error repeats in short unfamiliar contexts;
a pass would not distinguish memory effects from long-context/distractor effects
in the original example. The source and 167-token maximum input length are frozen.
No adaptation follows automatically, and the overall method goal remains unmet.

## Scope review before the next experiment

The current two-fact comparison is a development diagnostic. Its correction
scores are at a ceiling for all three representations, so it cannot establish
a correction advantage or validate the intended reusable behavior recipe.
The loss ablation and full-output review are complete. Do not
automatically follow it with another support/threshold variation on these same
facts. The next experimental proposal must connect its result to the library's
document-grounded answering application, use a strong common-objective LoRA
control, and separate supported answering, appropriate abstention and unrelated
utility. Existing failed QA pilots are evidence to audit, not fresh confirmation
data. Broad robustness and a second task family remain required for the goal.

The application audit rules out simply repeating the previous QA setup:

- [Evidence-pair v1](../experiments/evidence_pairs/colab_v1/REPORT.md) already
  tested carefully paired supported/missing contexts. The base answered31/32
  calibration missing cases correctly; all four supported exact-match misses
  conveyed correct content. Its twelve proposed fits were correctly skipped.
- [The dose diagnostic](../experiments/protected_rows/dose_v1/REPORT.md) found
  that improved refusal counts did not improve evidence discrimination; a
  constant refusal bias captured part of the benefit.
- [The external faithfulness probe](../experiments/faithfulness_probe/colab_v1/REPORT.md)
  exposed genuine interference, but dropping irrelevant context was already a
  strong baseline, and some source labels were questionable.

Therefore “return to grounded QA” is not yet an adequate experimental proposal.
A new development slice must require useful context, include similar but
nonanswering evidence, and establish semantic headroom after simple controls.
Use the existing paired runner where appropriate. Do not build another runner
or train on the exposed errors merely to obtain a positive result. This audit
does not establish which new slice will meet those conditions.

## What actually changes our decision

1. The [matched-budget study](../experiments/paired_mlp/complete_v1/REPORT.md)
   gets all8/8 requested transfer answers with selected, random and LoRA patches.
   Selected neurons cause unrelated Lakers and oxygen errors; chosen random and
   LoRA controls do not show those factual errors on this small bank.
2. [Earlier stopping](../experiments/paired_checkpoints/complete_v1/REPORT.md)
   keeps8/8 transfers and reduces selected training steps to12/12 versus30/24.
   Both factual errors survive. Total selected adaptation is3.83times LoRA's
   in that implementation, largely due to sparse attachment/export integrity work.
3. [Finite direction calibration](../experiments/functional_selection/complete_v2/REPORT.md)
   reduces one-step drift for similar learning gain but fails its numerical
   prediction criterion. It achieves no complete edits. Neither a diagonal risk
   sum nor a local base-softmax quadratic is a reliable finite-step certificate.
4. The [completed guarded-training study](../experiments/guarded_training/complete_v1/REPORT.md)
   gives3/8 transfer answers for risk-aware selection, random and LoRA. All six
   blocked fits' final rejected proposals improve task NLL but exceed worst-token
   KL.2. The other two fits stop with confident training answers yet fail a
   natural question form. Protection and transfer are separate problems.
5. The [saved-endpoint contract audit](../experiments/guarded_training/complete_v1/contract_audit.json)
   finds that the guard rejects all four older successful LoRA/random endpoints.
   At a valid short/long US alias branch, conditional KL exceeds16 while combined
   probability of covered acceptable answers remains above99.9%. This is a
   concrete false alarm, not a reason to erase genuine earlier interference.

The new guard's failure does not prove that no protected solution exists. It
shows that this guard is a poor proxy for the answer-quality requirement, and
that it prevented learning among the attempted steps. Some preservation labels
also disagree with the base: preserving a correct answer, correcting an existing
answer and forcing one preferred wording must not be silently conflated.

## Paper lessons that survive these results

| Source | Relevant mechanism | What we should take, and what we should not infer |
|---|---|---|
| [FiNE paper](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf), [pinned editor](https://github.com/opanhw/fine-unified-editing-code/blob/a2489cbac0dfe5a956385734065cfa041384e7d1/knowledge-editing/easyeditor/models/fine/fine_main.py) | Token-contribution localization plus composite loss, layer control and stopping | Localization, objective and optimizer form a recipe. Our all-layer gradient support/mean CE is not its reproduction. Original-prefix replication does not establish complete-answer transfer. |
| [NeST](https://arxiv.org/html/2602.16835v2), [code audit](../experiments/nest_source/README.md) | Harmful/benign probes, activation-profile clustering and shared incoming-row updates | A reusable behavior prior and shared capacity are plausible. Its balanced20k-example safety training is not a demonstration of three-example counterfactual fact editing. Full safety training remains unreproduced. |
| [SafeNexus](https://arxiv.org/html/2607.28969v2) | Cross-modal safety-neuron intersection and masked low-rank updates | Measure overlap with retained behavior. Activation exclusion alone does not establish update safety; the utility argument makes sensitivity assumptions. Official implementation is still unverified. |
| [SpaRTA](https://arxiv.org/abs/2502.15975), ordinary LoRA | Random sparse scalars and low-rank updates | Random and low-rank baselines need equal data, protection, tuning opportunity and actual-cost reporting. Whole-neuron random support is not a SpaRTA reproduction. |
| [Does Localization Inform Editing?](https://arxiv.org/html/2301.04213v2) | Empirical separation of causal tracing and best layer to edit | Knowing where a signal is causally important need not tell us where an update will generalize. Its layer-level findings are not a theorem against all neuron selection. |

Several tempting next ideas also have substantial prior art:

- [LoRA-One](https://arxiv.org/html/2502.01235v3) uses a first full gradient,
  spectral initialization and optional preconditioning. The
  [pinned implementation](https://github.com/YuanheZ/LoRA-One/blob/b797a10dcb818e9553b57fa060baa7b00df4484d/run_exp.py#L158)
  takes an SVD of the negative gradient and initializes a **nonzero** update;
  it explicitly skips offset subtraction for LoRA-One. A future comparison must
  count initialization's learning, discovery and cost. This is a stronger
  relevant baseline for our gradient-informed sparse approach, not a result we
  have already reproduced. Fixed-basis low-dimensional adaptation likewise has
  prior art; a tiny coefficient count can hide large stored directions.
- [ProFit](https://arxiv.org/html/2601.09195v3) masks tokens whose detached current
  probability falls below a threshold. Its one-to-many supervision concern is
  relevant. Its probability-as-importance heuristic is not justified for our
  counterfactual rare target names: masking their low-probability tokens can
  remove the correction signal we need. A lower-bound gradient argument also
  does not by itself order all actual parameter-gradient magnitudes.
- [EBFT](https://arxiv.org/html/2603.12248v2) trains against sequence-level feature
  moments using a frozen feature model and on-policy rollouts. It motivates
  judging completions rather than single teacher tokens. Importing its rollout
  estimator and second model now adds cost and another imperfect semantic proxy;
  it is not the minimal test for these short, explicitly enumerable answers.
- [Semantic loss](https://proceedings.mlr.press/v80/xu18h.html) already connects
  valid-output probability mass and learning. Summing probabilities of acceptable
  strings is ordinary marginal likelihood, not our proposed novelty.
- [PCGrad](https://arxiv.org/abs/2001.06782) is existing gradient-conflict surgery.
  Base-anchored KL has zero gradient at the base, so blindly projecting against
  that gradient cannot initially protect anything. Adam can also change a
  projected direction; acceptance must measure the actual proposed update.

FiNE's author-linked supplement could not be retrieved through the web reader;
its additional ablations are not claimed as inspected. Code inspection and
paper reading are distinct from an empirical reproduction. No upstream code
from these newly read projects was incorporated or executed.

## Candidate: constrain outcomes and coordinate the sparse update

This is a hypothesis to test, not a named new method or a recommended API.

For prompt x and an explicitly supplied finite set of complete EOS-ended
accepted responses Y(x), define

    p_good(theta,x) = sum[y in Y(x)] P_theta(y | x).

Duplicates count once; the probability includes every generated token and EOS.
The [experimental primitive](../experiments/guarded_training/response_sets.py)
computes this event exactly from all required prefix logits, with autograd.
Missing prefixes are errors. Its saved-logit diagnostic can report a covered
subset as an explicit lower bound. Even complete coverage of the supplied
finite strings does not certify all semantically acceptable responses.
An unenumerated valid punctuation/case/wording variant can still look like a
mass loss. Qualification must therefore include known benign output variations,
and probability checks cannot replace independent complete-answer review.

First separate correction and retention roles. Correction maximizes accepted
response mass on the desired behavior. Retention limits a decline in accepted
response mass from a base that actually supplies an accepted answer; it does
not preserve the relative probabilities of synonymous strings. Base/reference
disagreements require adjudication or an explicit correction role, rather than
quietly freezing the wrong target. Refusal/abstention is a separate outcome when
that task calls for it, not an interchangeable correct answer.

Then select **and coordinate** updates according to their joint correction and
retention effects. A possible next representation uses one cached task-gradient
proposal per neuron/group and jointly learns its coefficients under a trust
region. Joint sensitivity includes cross-component cancellation/reinforcement;
ranking independent risks loses that information. Local derivatives propose a
step, but actual accepted-response mass and complete answers accept it. A
full-vocabulary KL may remain a diagnostic; it must not silently define all
behavioral damage. A no-edit result is permitted and is not adaptation success.

The coefficient representation, clustering and constrained optimization all
have precedents. Its value would have to be **better held-out correction versus
preservation for less data or lower actual total cost**. If the corrected
objective gives the same benefit to ordinary LoRA, that is objective repair,
not a selected-neuron research win. Compare random supports, ordinary LoRA and
a gradient-informed low-rank baseline with the same protection contract; include
source-data, validation, initialization, storage and discovery costs.

## Completed qualification and the next causal question

The [completed outcome-guard replay](../experiments/outcome_guard/complete_v1/REPORT.md)
restores six original endpoints and two bases, independently verifies840 prefix
vectors and96 generated observations, and preserves the frozen0.02 criterion.
The all-positive-control hypothesis **fails**: only two of four previously
successful LoRA/random controls meet the probability limit. Both known selected
Lakers/oxygen errors are flagged. All six endpoints still reproduce their
requested4/4 transfers. No new parameters were fitted.

The rejected controls are not simply harmless alias changes. Conditional on O,
the probability of continuing with ₂ is0.00307% in the base,8.87199% for random,
18.85583% for LoRA and95.53709% for the selected case1 patch. Random and LoRA
still greedily answer O. These are next-token probabilities, not measured sampled
whole-answer error rates. Calling those older controls universally clean was
too strong; their apparent success was based on observed greedy answers.

The finite-set event ignores much of the benign short/long US redistribution
that inflated conditional KL, while revealing real competition on oxygen. Keep
both correct complete answers and accepted probability as separate measured
requirements. Do not loosen the frozen threshold to turn this failure into a
pass, or remove confidence reporting merely because the greedy choice is correct.
The [post-hoc trie audit](../experiments/outcome_guard/complete_v1/decision_audit.json)
also agrees with cached generation on all64 observed traces. A probability-tree
counterexample shows why event mass alone cannot replace greedy-answer checks.

There is a concrete auxiliary-loss mismatch to isolate next. The
[logit-gradient audit](../experiments/outcome_guard/complete_v1/objective_audit.json)
finds that canonical US labels teach the base to suppress EOS after its already
accepted United States answer. Mean-token CE gives an EOS-logit gradient near
+0.2 there; an accepted-set marginal loss with the **same canonical-length
normalization** gives gradients near−1e−7. This comparison identifies unnecessary
stopping pressure at those logits. It does not yet establish that those
auxiliary gradients caused the unrelated O₂ mistake.

Run a matched objective ablation before changing supports again: keep the same
original examples, rates, supports,50-update schedule, sampling order and length
normalization. Change only supervision on the five initially accepted auxiliary
rows per case from one canonical response to the existing finite accepted set.
Keep the two unresolved base/reference examples explicitly identified and
identical across arms. Leave oxygen and Lakers outside the training bank and
measure their complete answers, accepted mass and competing continuations.
The verified original canonical endpoints are reference controls only if the
new implementation recovers that control exactly. This is a causal diagnostic,
not a claim that the failed protection rule has now qualified.

If removing that pressure improves every representation, it is objective repair
with potential library value, not a selected-neuron advantage. If it does not,
correction-gradient directions and update coordination remain leading questions.
A later selector can measure **decision-margin sensitivity** rather than relying
only on Fisher risk, which downweights currently improbable wrong tokens; this
is a hypothesis requiring its own comparison and actual finite-step checks.
Preservation examples must cover the relevant failure modes. Local ranking,
neuron labels and a small update norm do not provide that coverage automatically.

The [joint-update candidate note](joint-update-candidate.md) separates the
outcome objective, coefficient representation and controller, and relates them
to GEM, A-GEM and GPM. Joint constraints and fixed-basis adaptation already have
prior art. Count stored directions and total cost, not just coefficient count.
Neither that proposal nor this diagnostic establishes superiority over LoRA.

A trained advantage still requires fresh cases, several parameter/data budgets,
additional seeds, another task family and model-size checks after freezing the
recipe. Keep validation used for selection out of confirmation claims. The
original full method/library goal remains active and unachieved.
