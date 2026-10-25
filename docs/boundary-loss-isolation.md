# Isolate the shared regression before changing the selector

Status: diagnostic design, not submitted. This is a follow-up to the
[completed objective comparison](../experiments/instruction_boundary/complete_preference_v1/REPORT.md).
The exposed 30-document evaluation is now development evidence. It cannot be
reused as fresh confirmation of a repaired method.

## Question and interventions

For the existing diverse-SFT run, the optimized loss was

\[
L(\theta)=L_{\rm task}(\theta)+L_{\rm short\ reference}(\theta).
\]

Each component is mean completion-token negative log likelihood, including EOS.
The second term can change a correct response distribution: being a correct
reference answer does not make its gradient zero at the original model.
Both components favor short answers, so the observed response shortening does
not distinguish them. The observed failures also need not be caused by length.

Run only two new fits, both from the same base and initial adapter:

1. **Preservation only:** optimize the same 31 short references, same order,
   186 steps, with no task gradient.
2. **Task only:** optimize the same 186 diverse chosen completions, same order,
   186 steps, with no preservation gradient.

Compare with the existing untouched base and combined diverse-SFT artifact.
Keep model revision, adapter architecture/seed, learning rate, optimizer,
precision, clipping and decoding identical. Reconstruct both sample permutations
before dropping either objective so random-number consumption cannot change
sample order. Verify the initial adapter identity and replay the existing
combined adapter on the diagnostic cases before attributing differences to loss.
The earlier report supplies no optimizer-path equivalence for these new arms.

Do not add a KL repair or tune a learning rate in this run. Those changes would
answer a different question before the cause has been isolated. Removing a loss
changes gradient scale and Adam's trajectory; interpret each ablation as the
causal effect of removing that term at this recipe, not a universal indictment
of the objective. Nonadditivity of outcomes is possible.

## Evaluation and stop rules

Use all 30 exposed clean cases, the 60 held-out-wording competing-question cases
(injected and legitimate), and 30 authority cases at the final checkpoint.
Record clean outputs at steps 31 and 93 as trajectory diagnostics; do not select
a checkpoint using these results. Preserve complete responses, token counts,
termination, per-example losses, gradient norms, runtime and adapter bytes.
Review new unique responses with arm identities hidden, using the existing
source rubric, and freeze decisions before attribution.

Primary diagnosis is the *identity* of base-correct clean regressions, especially
the six shared failures. Report all clean gains/losses, not just the six examples.
If preservation-only reproduces errors, that term is sufficient to damage those
answers at this recipe. If task-only does so, task training is sufficient. If
neither alone does, investigate combined optimization; do not claim either term
is harmless. If both do, replacing preservation alone is unlikely to suffice.
The exact number of reproduced failures is evidence, not an all-or-nothing gate.
A shortening-only result without errors weakens the claim that brevity itself is
the cause. New failures matter even if the original six improve.

The existing qualification gates remain the standard for a useful behavior;
this diagnostic cannot establish a better method or a selector advantage.
At most two fits in one private free Kaggle job, sequentially; no full model on
the Mac, no automatic resubmission. Freeze executable/data hashes and an explicit
runtime cap after tiny-model validation and before submission. Stop on identity,
numerical, truncation or runtime violations and retain partial evidence.

## Connection to the library goal

A useful selector must find parameters that can learn the intended change while
preserving useful behavior. Ranking neurons cannot repair contradictory or
unnecessarily restrictive training targets by itself. This ablation tells us
which learning signal a future selection method would need to accommodate.
It does not imply that successful adaptation always requires passing this one
prompt-injection benchmark, or that this table task should define the product.
After isolating the regression, evaluate the repaired recipe against ordinary
LoRA and matched random neuron supports on a fresh split and a second task.
Only a measured quality/preservation/cost advantage justifies a library default.
