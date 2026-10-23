# Current method decision — 23 October 2026

**The goal remains unachieved:** a robust sparse-adaptation library with a
verified practical advantage over strong baselines. Compact patches, training,
export/reload and evaluation work. No recommended neuron-selection recipe or
superiority claim is established. LoRA remains the practical comparison baseline.

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
are now exposed. No additional GPU job is running or submitted. Do not launch
another format or support sweep on this test bank.

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

The [experiment history](experiment-history.md) preserves the earlier work.
Historical candidate methods are hypotheses, not approved next runs.
