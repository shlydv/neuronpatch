# Experiment history

These notes were moved from the README to keep the entry point focused on use.
They preserve the previous evidence narrative and its report links. For the active
method, pending experiment and present decision, use [current-method.md](current-method.md).
The individual reports define each result's model, data and evaluation scope.

**Status: experimental foundation.** Version 0.1 implements a tested PyTorch
weight-update primitive, not a validated behavioral editing system. Current experiments
expose correction/preservation trade-offs. Our custom QA experiments have not shown
a guided-selection advantage. A [16-case upstream FiNE reference](../experiments/fine_reference/reference16_v1/REPORT.md)
gets the requested prefix on 15/16 original prompts versus 0/16 for random neurons,
but only 5/16 rewordings. We have not demonstrated an improved method or a
recommended recipe. See the [evidence ledger](method-design.md).

The [first protected-row Qwen3-4B comparison](../experiments/protected_rows/colab_v1/REPORT.md)
is complete: protection shows a partial correction/preservation trade-off improvement,
but all eight settings fail the declared joint success criteria. Output review finds
real unnecessary refusals as well as noisy exact-match labels. The experimental
constraint is not part of the public API or a recommended behavior recipe.
The [dose/bias diagnostic](../experiments/protected_rows/dose_v1/REPORT.md) verifies
exported generations exactly and finds no measured evidence-discrimination gain;
part of the improvement comes from shifting the refusal trade-off.

The [external faithfulness probe](../experiments/faithfulness_probe/colab_v1/REPORT.md)
finds real reference-answer interference, plus scoring and source-label problems.
Ignoring the untrusted passage is already a strong baseline in its factual mode.
No new trained method is promoted by that diagnostic.

A [two-case GPT-J transfer pilot](../experiments/fine_transfer/pilot_v1/REPORT.md)
finds a positive selection effect: cross-prompt update gradients yield 6/6 requested
answer prefixes on untrained wordings versus 3/6 for FiNE supports given the same
training data. Its [frozen subject-scope follow-up](../experiments/fine_transfer/scope_v1/REPORT.md)
then finds the requested target spreading to 9/16 other subjects, including all
eight people in the citizenship case. Original FiNE also leaks on 10/16 probes.
The transfer gate passes, but subject specificity fails. No current experimental
selector is recommended as a behavior-editing recipe.

The [fixed-support preservation experiment](../experiments/subject_binding/pilot_v1/REPORT.md)
reduces unseen-subject leakage from 8/8 to 4/8 while retaining 6/6 requested
untrained-wording prefixes. Calibration succeeds, but the declared scope gate
fails and full continuations expose additional interference. This is a useful
partial result, not a validated adaptation recipe or superiority claim.

The completed [common-objective comparison](../experiments/adapter_comparison/pilot_v2/REPORT.md)
favors broad official LoRA: 6/6 untrained-wording starts and 0/8 replacement
leaks, versus 5/6 and 1/8 for simpler mean-gradient sparse support, and 6/6 and
4/8 for saved consensus support. LoRA uses 50–200 times more trainable scalars;
this is not a parameter-matched efficiency result. All 530 capped generations
are verified and reviewed. Correct prefixes still hide later errors, and the
weak base makes utility retention an insufficient quality check. These are two
exposed development facts, not a validated recipe. The
[complete frozen-adapter group diagnostic](../experiments/adapter_diagnosis/complete_v1/REPORT.md)
verifies18 interventions and1060 outputs without refitting or consuming fresh
facts. Paired MLP updates preserve6/6 requested starts but still produce bad
tails. Final-three-only updates give0/6; excluding them retains6/6. This narrows
the next design, not a validated recipe.

The [experimental paired-MLP implementation](../experiments/paired_mlp/README.md)
now supports matching incoming rows and outgoing columns with FP32 deltas,
complete-response supervision through EOS, compact reload and merge checks.
The [completed matched-budget Qwen3-4B pilot](../experiments/paired_mlp/complete_v1/REPORT.md)
verifies24 fits and1417 observations. Selected paired, random and LoRA conditions
all give8/8 complete transfer answers on two development facts. Gradient-selected
paired updates also introduce unrelated errors; no selection advantage or robust
default recipe has been demonstrated. Full review distinguishes valid aliases
missed by exact scoring from genuine interference.
The [completed zero-fit endpoint replay](../experiments/paired_endpoint/complete_v1/REPORT.md)
checks496 answers and finds all248 live/merged pairs identical. The observed
failures already exist before merging; deployment rounding is not their cause
on these tested endpoints.
The [completed checkpoint study](../experiments/paired_checkpoints/complete_v1/REPORT.md)
verifies seven fits and434 answers. Guided supports fit in12 updates versus30/24
for random and LoRA, but their two unrelated factual errors persist. Early LoRA
retains all8/8 transfers without measured preservation losses and has a shorter
total adaptation prefix. A late high-rate failure is isolated to an optimizer
update causing premature EOS; it is distinct from the earlier semantic errors.
The current guided recipe does not advance unchanged to fresh confirmation.
