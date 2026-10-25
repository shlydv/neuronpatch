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

## Implementation checkpoint — 25 October 2026

The shared optimizer now supports explicit `task_only` and `preservation_only`
controls. It constructs both seeded permutations in all modes and skips disabled
forward/backward passes. The default combined SFT/DPO path remains available.
Per-step records identify active losses, skipped losses as null, actual forward
counts, gradient norms and retries. A callback can observe successful updates
for the predeclared intermediate diagnostics.

The experiment runner supports the two controls and retains frozen-base checks,
compact export, exact score equality after reload and evaluation of the reloaded
artifact. Eleven focused CPU tests pass, including independently computed Adam
updates for both controls, sample-order equality, the two-arm tiny-Qwen lifecycle,
the original four-arm lifecycle, and the half-precision worker. This verifies the
implementation on small models, not the full-model scientific hypothesis.

Remaining before submission: wire the diagnostic evaluation subset and checkpoint
callback, package the old combined adapter for exact output replay, bind the
runtime/source/data identities, and test the resulting worker end to end.

The diagnostic worker is now implemented and covered by a half-precision tiny-Qwen
end-to-end test: it reloads the prior combined adapter, checks prompt IDs, generated
token IDs, decoded text and termination against saved outputs before fitting,
then completes both controls and intermediate generation. A changed replay token
is rejected. Eleven focused tests pass after this addition.

[Protocol v1](../experiments/instruction_boundary/loss_isolation_protocol_v1.json)
fixes 120 final diagnostic cases, clean outputs at steps 31 and 93, two fits and
480 generated records including replay. Runtime limits are 7,200 seconds at the
provider, 6,900 for the worker and 6,600 internally. The old adapter and outputs
are hash-bound and supplied through the previous private Kaggle kernel's output;
no model checkpoint or source table is published in Git. This protocol supersedes
the earlier pending-implementation checklist above.

Private Kaggle submission `shlydv/neuronpatch-loss-isolation-v1-20261025`
(version 1, kernel 140741997) was accepted on 25 October 2026 and observed RUNNING.
Accepted source matches local SHA256
`24db77e989739adca7816b0d9354a193f6b58814ef9769cdb23443170a9deb9f` exactly.
The previous preference job was verified COMPLETE before submission. Dependency
setup reached the worker launch; replay and training results are not available
at this checkpoint. Local read-only collection is attached to this same job;
observation interruptions must not trigger a replacement run.

A separate CPU verifier now checks the downloaded loss-control artifacts against
frozen inputs, reconstructs both sample permutations, checks active versus skipped
loss records, counts actual forward calls including retries, validates finite
exported parameters and checks prompt/token/text consistency for all final and
intermediate generations. It rechecks the prior adapter hash and exact output
replay against the original artifact. The tiny-model end-to-end test passes and
rejects a deliberately introduced non-null disabled preservation loss. These are
artifact checks, not an independent rerun of GPU training or semantic grading.

## Why correct-reference training is not distribution preservation

Rechecked against the [FiNE paper, equations 4–7](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf)
and [official knowledge-editing code, lines 206–212](https://github.com/opanhw/fine-unified-editing-code/blob/main/knowledge-editing/easyeditor/models/fine/fine_main.py).
The paper combines target likelihood, KL and a repetition penalty. The code uses
worst-token target NLL, a norm term, prefix-distribution KL and a repetition term;
its KL direction is reference-to-edited, unlike the paper's written direction.
This is a scoped editing recipe, not a demonstration on our table instruction task.

Our mathematical implication: for one teacher-forced token with logits z,
reference distribution q and current distribution p, reference-answer NLL has
logit gradient p − one_hot(y). Even at the untouched model, this generally is
nonzero. A correct y does not change that. Training on the base model's greedy
answer also has this property; hard-label self-distillation is not a no-op.
By contrast, KL(q || p) has gradient p − q, zero when p = q at initialization.
It can oppose later distribution drift, but cannot guarantee retained accuracy
on unobserved contexts or remove conflicts with the desired task update.

For the library, these should eventually be explicit, distinct policies:
reinforce a reference answer versus retain a reference distribution. Neither
should be silently described as guaranteeing preservation. Do not implement or
launch a KL repair merely because this derivation is appealing: first finish
the current causal diagnostic, then test any repair against a competent LoRA
baseline and matched sparse controls. Prefix KL from FiNE and response KL on
preserved tasks are different constraints; specify token positions and direction.

## Result and implementation direction after the ablation

The [completed report](../experiments/instruction_boundary/complete_loss_isolation_v1/REPORT.md)
shows that task-only reproduces all six combined clean regressions, while
preservation-only reproduces five. The shared pressure is not isolated to the
auxiliary term, and removing it worsens task-trained clean accuracy.

`response_retention.py` now adapts the existing library KL primitive to explicit
response trajectories: it shifts target masks once to prediction positions,
caches only those full-vocabulary reference logits on CPU, binds the sample and
base identities, and rejects trainable teachers, changed trajectories and cache
budget overflow. It is experiment-local, not a new public loss or default.
Three focused CPU tests verify causal positions including EOS, zero loss/gradient
at an unchanged model, agreement with an independent soft-target cross-entropy
gradient after perturbation, and lifecycle/identity errors.

This helper deliberately does not choose which trajectories to preserve. KL on
a short manually written answer still measures conditionals along that forced
prefix, not along the base model's usual longer answer. The next experiment must
specify that choice; otherwise a changed target style and a changed objective
would be confounded. Full-response retention on base-generated clean trajectories
is a candidate, but it can also preserve base mistakes. Any selected trajectory
review must use training/calibration only. No new GPU run is submitted here.
