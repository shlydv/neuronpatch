# Current method decision — 24 October 2026

**The goal remains unachieved:** a robust sparse-adaptation library with a
verified practical advantage over strong baselines. Compact patches, training,
export/reload and evaluation work. No recommended neuron-selection recipe or
superiority claim is established. LoRA remains the practical comparison baseline.

**Earlier result, 24 October:** the [math baseline](../experiments/relation_learning/math_baseline_complete_v1/REPORT.md)
completed and passed independent token/artifact verification. Review yields124
accepted answers, one completed mathematical error and three incomplete responses
out of128. Two reference answers are mathematically wrong; the model is correct
on both. The preset sixteen-error qualification fails, so no math adaptation
run follows. Kaggle consumed2040.32 provider seconds. The active boundary
comparison is described below; this math experiment is finished.

The next [candidate task](instruction-boundary-candidate.md) tests instruction
boundaries while preserving useful table QA. A pinned source/data audit found
that BIPIA's default builder embeds context in the system role and that its email
slice is mostly unanswerable. The table training slice has900 questions over617
contexts. The [data protocol](../experiments/instruction_boundary/README.md) is now
frozen with128/32/64 document-disjoint train/calibration/development tables and
seven cases per table. All1568 examples pass tokenization checks. Six CPU tests
cover construction, paired qualification, inference and independent verification.
The [completed baseline](../experiments/instruction_boundary/complete_v1/REPORT.md)
ran448 development generations with zero training updates. All128 injected
responses return the attacker marker; all256 legitimate direct/quoted controls
pass. Review establishes43 completed clean-correct documents, all harmed by both
injection families, plus11 wrong, three source-ambiguous and seven incomplete
clean responses. Two wrong references were corrected against their tables.
The original qualification fails its48-clean-correct requirement; the other four
checks pass. This is real paired behavioral failure, not a trained-method result.
No automatic training or retroactive threshold change follows. A subsequent
explicit exploratory protocol can investigate preservation of existing correct
behavior without conflating it with learning table reasoning. The run is COMPLETE,
source/token/manifest verification passes.

The explicit [exploratory follow-up protocol](../experiments/instruction_boundary/transfer_protocol_v1.json)
now has32/16/48 fresh training/calibration/confirmation documents, excluding all224
previously allocated tables. It transfers clean-teacher answers to injected
prompts, preserving clean/direct/quoted responses. A NeST-inspired probe/grouped
candidate and two random-group controls each train368,640 scalars; rank1 LoRA
across attention and MLP trains2,064,384. All receive the same targets,96 Adam
steps and two rates. Only redirect examples enter fitting and rate selection;
claimed authority is held out. Source and numerical distinctions from NeST are
explicit, and no method advantage is assumed.

Kaggle accepted kernel140449387 version1 on24 October and reports RUNNING. The
[accepted source](../experiments/instruction_boundary/transfer_submission_accepted_v1.json)
matches the tested frozen bundle hash. Provider/worker/internal caps are
10800/10200/9900 seconds; paid scaling is disabled. Twenty-nine relevant checks
passed, including an eight-fit tiny FP16 Qwen lifecycle and verifier corruption
fixtures. The real-model run is pending; these software tests are not quality
evidence. Observe this same job rather than starting another if logs disconnect.

### Interpretation audit before the comparison finishes

The current selector distinguishes clean table prompts from the same tables with
an appended redirect instruction. It does not distinguish an unauthorized
instruction from a legitimate request containing the same words: direct and
quoted controls enter fitting and calibration, but not probe discovery.
Consequently, high probe accuracy could reflect instruction wording, marker
presence or prompt length. It cannot by itself establish a useful editing target.
The frozen claimed-authority test changes attack wording, but retains the marker,
append location and table format. Even success on that family would establish
limited transfer, not general prompt-injection resistance.

Our [NeST reference](https://arxiv.org/html/2602.16835v2) supports studying whether
activation clusters also have aligned training gradients (Section V). It does
not establish that this property holds for our task. We deliberately fixed two
groups per projection and 32 selected rows, whereas NeST chooses clusters using
activation geometry. This run therefore tests the declared adaptation, not the
paper's full method or its strongest possible configuration.

Read the pending evidence in three stages: does training recover correct answers;
does it preserve legitimate instructions and clean answers; and does guided
selection improve on both random supports at the same budget? Only then compare
quality, storage and measured costs with LoRA. The gradient diagnostics can help
explain a result, but two training examples and two calibration documents do not
establish population-wide alignment. If selected and random supports perform
similarly, do not interpret compactness as a selection advantage. If all methods
fail, inspect shared targets, learning curves and output errors before blaming
neuron localization. These are interpretation rules, not changes to the frozen
experiment or permission to tune on confirmation outputs.

Takeover review: the current Fisher ranking does not implement NeST's probe,
clustering or shared update method. NeST SectionV additionally measures gradient
alignment within versus between activation-derived clusters. That supplies a
testable connection between discovery and trainable capacity, not a proof that
our proposed behavior task has the same structure. The next method decision must
test whether update directions agree across training examples and transfer to
separate calibration examples while preserving legitimate controls. Large squared
gradient norms alone cannot answer that question. The existing actual-update
observer is instrumentation for this investigation, not an established selector.

## Latest completed evidence

The [relation-learning pilot](../experiments/relation_learning/complete_v2/REPORT.md)
uses Qwen3-4B-Instruct-2507, 64 task labels, 20 preservation labels, 64 updates and
approximately matched trainable-scalar budgets. Rates were chosen from validation
loss before reviewing held-out outputs.

| Validation-selected condition | Task /64 | Solved-family controls /16 | Utility /8 |
|---|---:|---:|---:|
| Base model | 39 | 16 | 8 |
| LoRA | 64 | 16 | 8 |
| Random seed 43 | 64 | 16 | 8 |
| Random seed 71 | 64 | 16 | 7 |
| Empirical-Fisher neurons | 63 | 16 | 8 |

The task is learnable: training corrects both unsupported guesses and unnecessary
refusals on fresh names and changed wording. Fisher adds one unnecessary refusal;
random seed 71 breaks a binary conversion. The unselected high-rate Fisher fit
returns empty responses on all eight utility questions. These controls are small
and synthetic; retaining them is not broad preservation evidence.

All eight fits and 2,880 outputs pass independent artifact verification. A bounded
loss-scale repair reproduces the earlier successful LoRA fit exactly, including
adapter tensors, teacher tensors, generated tokens and common training history.
The completed v2 job consumed 2,908 provider seconds. It establishes learning,
not a new adaptation method or full reproduction of the cited papers.

The [saved-score audit](relation-score-decision.md) shows Fisher captures 43.25%
of its ranking score versus about 0.02% for random, without better trained quality.
The balanced support already attains 97.86% of the global score-only optimum.
Improving this proxy or reallocating layers is not the next experiment.

## Latest low-data result and next decision

The [eight-task-example pilot is complete](../experiments/relation_learning/complete_low_data_v1/REPORT.md).
With twenty additional preservation labels and the same 64 updates and parameter
budgets, validation-selected LoRA scores 126/128; Fisher 121/128; random seeds
43 and 71 score 112/128 and 122/128. All four score 64/64 on original wording;
all remaining errors occur under transfer wording. Random seed 43 additionally
loses one of eight utility answers. All four retain sixteen scope controls.

The selection-advantage criterion fails. The all-baselines-near-perfect ceiling
criterion also does not trigger; do not rewrite either criterion. The unselected
high-rate Fisher fit scores 56/128 with substantial control damage. Every fit is
numerically finite with zero retries. Independent checks verify all 3,024 outputs
and eight teacher/adapter records; Kaggle completed in 2,660.59 seconds.

The recorded sparse condition cycles are about twice LoRA's, including export,
evaluation and restoration; this is not a pure training-speed measurement.
Discovery adds 34.42 seconds. No data or actual-cost advantage is demonstrated.

Original-wording validation loss does not reliably select transfer performance.
One unselected random rate scores 127/128, but replacing the frozen selection
with that test winner would be invalid. The next method comparison must match
validation to intended deployment and use a broader task; these reviewed inputs
are now exposed. No additional relation-learning GPU job is running or submitted.
Do not launch another format or support sweep on this test bank.

The [semantic-review tool](../experiments/relation_learning/adjudicate.py) binds
explicit nonexact judgments to inputs and responses, then computes paired
corrections and damage. It is bookkeeping, not an independent semantic judge.
The [token audit](../experiments/relation_learning/complete_low_data_v1/name_fragment_audit.json)
also prevents mislabeling Fisher's “I” for Idris as pure truncation: it changes
the first token and then emits EOS. Its internal cause remains unknown.

## What we have ruled out or learned

- Earlier two-fact edits reached a ceiling for all representations while selected
  neurons caused unrelated errors. Earlier stopping did not remove them. See the
  [matched study](../experiments/paired_mlp/complete_v1/REPORT.md) and
  [checkpoint study](../experiments/paired_checkpoints/complete_v1/REPORT.md).
- Token-distribution preservation can penalize valid aliases. Accepted-answer
  probability and complete-answer correctness must be reported separately.
  [Outcome checks](../experiments/outcome_guard/complete_v1/REPORT.md) and the
  [completed objective ablation](../experiments/alias_supervision/complete_v1/REPORT.md)
  identify real issues but do not establish a selected-neuron advantage.
- The [retrieved-evidence slice](../experiments/retrieval_evidence/complete_v1/REPORT.md)
  lacked sufficient repeated errors for adaptation. The
  [precision diagnostic](../experiments/inference_sanity/complete_v1/REPORT.md)
  did not attribute the observed errors to the tested device/precision settings.
  Neither warrants another generic prompt or dtype sweep.
- Causal localization, deletion sensitivity, gradient magnitude and useful finite
  training updates are different interventions. A selector must predict the one
  we actually apply. Raw gradients also depend on parameter coordinates; see the
  [coordinate audit](selection-coordinate-audit.md).

## Paper synthesis and remaining requirements

[Research provenance](research.md), the [23 October review](research-2026-10-23.md),
[benchmark design](adaptation-benchmark-decision.md), and the
[archived detailed decisions](method-history-2026-10-23.md) retain source links,
code audits, limitations and failed hypotheses. FiNE combines localization with
an objective and optimization recipe; NeST's behavior training is not a few-fact
editing result; our grouped Fisher/random supports do not reproduce their full
methods. SafeNexus's official implementation remains unverified. Selection with
LoRA, gradient-informed initialization and constrained updates have prior art.

A useful contribution must show better held-out quality/preservation at lower
label, parameter, storage or actual compute cost, compared fairly with tuned
LoRA and random supports. Promising synthetic results still need another task,
additional seeds, meaningful unrelated-task coverage and a model-size check.
Validation cannot double as confirmation. No new named method, default API or
claim of beating prior work should precede that evidence.

The [saved-update geometry audit](update-geometry-decision.md) now examines all
twelve paired patches from the two completed relation runs. A single shared row
captures only 20.3–22.9% of update energy; the best signed rank-one approximation
captures 37.7–44.9%. This does not measure behavioral compression or refute NeST
clustering. It removes the assumption that these successful trained patches are
already mostly one reusable direction. No new GPU run or default method follows.

An [actual-update prediction observer](update-prediction-contract.md) now previews
compact Adam/AdamW steps without changing live training state and compares signed
calibration-loss predictions against measured changes. Six CPU checks include
bit-identical reference-loop training and distinguish momentum mismatch from
finite-step error. GPU/retry integration and real-model predictive utility remain
unverified; this is instrumentation, not a selected-neuron advantage. Retry-safe
observation is now tested with real FP16 overflow: measurements occur once per
accepted step, without changing training; exhausted retries produce none.

The next broader [development slice](../experiments/relation_learning/math_development_v1/README.md)
contains 288 deterministically partitioned GSM8K training examples. Source hashes,
license, disjoint memberships and pinned-tokenizer preflight are saved. The
official test set remains untouched. Its first stage is unmodified-model baseline
qualification, not another selector fit. That baseline is now complete and fails
qualification, as summarized above. The saved development data must not be
silently repurposed as a new confirmation benchmark.

The [experiment history](experiment-history.md) preserves the earlier work.
Historical candidate methods are hypotheses, not approved next runs.
