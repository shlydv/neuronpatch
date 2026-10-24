# Current method decision — 24 October 2026

**The goal remains unachieved:** a robust sparse-adaptation library with a
verified practical advantage over strong baselines. Compact patches, training,
export/reload and evaluation work. No recommended neuron-selection recipe or
superiority claim is established. LoRA remains the practical comparison baseline.

**Current status:** boundary-transfer v2 is COMPLETE, returncode0, wrapper elapsed
6527.17 seconds. [Independent artifact verification](../experiments/instruction_boundary/complete_v2/integrity.json)
passes for1728 generations and eight fits. All fits have zero gradient retries;
peak allocated GPU memory is8.28–8.31 GiB. Baseline generations, teachers,
supervision, supports, probes and group geometry exactly replay the failed v1
run. No GPU experiment remains active; semantic review is complete.

The [completed answer review](../experiments/instruction_boundary/complete_v2/REPORT.md)
establishes27 unambiguous base-clean-correct documents. LoRA recovers25 under
redirect and21 under claimed authority; guided groups recover24 and0; random43
recovers25 and19; random71 recovers25 and4. LoRA and random43 preserve all27
clean answers. Guided groups introduce2 clean regressions and random71 introduces3.
Five ambiguous QA documents are excluded consistently across arms; all192
legitimate direct/quoted controls per model pass. Strict supporting-claim and
lenient answer-identity sensitivity analyses preserve the method conclusion.
Guided selection fails the frozen learning, selection, LoRA-quality and
preservation criteria. Compact adaptation works in this pilot, but no robust
selection advantage or training-speed advantage is established.

### Next investigation: authority-matched discovery

The experimental [pair constructor](../experiments/instruction_boundary/authority_pairs.py)
now creates two prompts with the same question, table and redirect wording.
The redirect appears inside the quoted JSON document in one and outside it as
the user's final instruction in the other. Unlike the previous clean-versus-
injected discovery labels, both classes contain the payload and its marker.
Only the32 original training documents enter; generated answers and QA references
do not. CPU tests check the decoded document boundary, identical payload content,
source immutability and malformed-input rejection. The tokenizer-only
[manifest](../experiments/instruction_boundary/authority_pairs_v1.json) records
length differences rather than asserting the prompts are token-identical.

This tests a specific suspected shortcut. It does not supply a new training
objective, prove that the probe identifies editable neurons or establish general
authority understanding. A causal comparison must keep the patch budget,
grouping, training targets, rates and optimization fixed while changing discovery
examples, and evaluate on fresh documents plus previously untested attack
wordings. The existing confirmation set is exposed and must not choose the new
support or rate. Existing trained comparator adapters can be evaluated on those
same fresh inputs rather than retrained unnecessarily.

[IH-Challenge](https://arxiv.org/abs/2603.10521) explicitly separates hierarchy
failures from difficult instruction following and warns about shortcut learning.
[SecAlign](https://arxiv.org/abs/2410.05451) contrasts legitimate and injected
responses using preference optimization. These motivate controlled contrasts;
neither establishes the effectiveness or novelty of this neuron-selection idea.
Implementation status: the worker and artifact verifier now support matched
pairs. A real tiny-Qwen training fixture covers both discovery modes, adapter
reload, merged evaluation and exact base restoration. Frozen comparator replay
checks every saved adapter file against a hash manifest, requires the original
base fingerprint and performs no training. The test verifies its merged model
fingerprints equal the original evaluation fingerprints and rejects corrupted
adapter bytes. Text sampling is stubbed in this CPU fixture; it does not prove
real-model quality or generation equivalence.

The fresh private confirmation bank contains48 documents (528 cases), excluding
all320 previously allocated documents. Training and calibration cases remain
unchanged. It adds one previously untested grading-notice wording and a
full-document legitimate-instruction control. Its input SHA256 is
`65fb4a1db3a4313f4e07ada8b8848056bd990af2a6547f36f10fd109b1272cd0`.
The parent adapter manifest is [pinned here](../experiments/instruction_boundary/authority_parent_v1.json).
Worker integration and local preflight are complete. The [frozen protocol](../experiments/instruction_boundary/authority_protocol_v1.json)
requires two new fits and four saved comparator endpoints; all3216 planned
teacher/confirmation generations share the prior limits. The tokenizer preflight
finds maximum prompt1018 tokens and training sequence1101, within frozen caps.
Seven tests cover training, replay identity and corrupted/missing artifacts.
The private [Kaggle run](https://www.kaggle.com/code/shlydv/neuronpatch-boundary-authority-v1-20261024)
was accepted as kernel140580989 version1 and is RUNNING at the initial status
check. The downloaded accepted source matches the frozen SHA256. Provider cap
is10800 seconds; paid scaling is disabled. No outcomes are available yet.

The [source-only audit](../experiments/instruction_boundary/authority_source_review_v1.json)
was frozen before inspecting this run's responses. All48 questions were reviewed:
three references need correction (indices6,9,45), one permits additional valid
pairs (46), and one lacks the required nationality data (14), so its QA is
excluded for all arms while controls remain. Index43 has a predeclared sensitivity
for distinct tournament names versus appearances. These are assistant judgments,
not independent human review. The private audit is hash-bound to the inputs.
The new review packet withholds condition names and keeps attribution separately;
it rejects stale output bindings, incomplete coverage and successful scores for
unterminated answers. These review tools do not change the running worker.

The [raw marker outcome](../experiments/instruction_boundary/complete_v2/preliminary.json)
also rejects the guided candidate's transfer criterion. All trained arms stop
returning the attack marker on the48 redirect cases. On48 claimed-authority cases,
guided selection still returns the attacker marker48 times, versus18 for LoRA,
20 for random43 and43 for random71. All192 legitimate direct/quoted requests per
model still return the requested marker. These are exact response counts, not
semantic table-answer accuracy. Guided claimed-authority accuracy is necessarily
zero because every response is the unrelated marker; it cannot beat either
random arm by four correct answers. The response-bound review above additionally
distinguishes actual recovered answers from merely suppressing the marker.

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

Kaggle accepted kernel140449387 version1 on24 October. It subsequently ended
with an out-of-memory error, not a method-quality result. The
[accepted source](../experiments/instruction_boundary/transfer_submission_accepted_v1.json)
matches the tested frozen bundle hash. Provider/worker/internal caps are
10800/10200/9900 seconds; paid scaling is disabled. Twenty-nine relevant checks
passed, including an eight-fit tiny FP16 Qwen lifecycle and verifier corruption
fixtures. Baseline generation and discovery completed, but the first LoRA fit
failed before its first optimizer update: the calibration forward inside the
pre-update observer could not allocate another166 MiB on the14.56 GiB GPU.
The wrapper records889.19 seconds and returncode1. All available artifacts have
been downloaded; no completed training comparison exists.

The repair under test adds non-reentrant decoder activation recomputation while
preserving eval mode, frozen embeddings and the observer's autograd.grad path.
It is opt-in and does not alter the archived v1 protocol or results. Tiny-model
checks cover both ordinary and recomputed eight-fit lifecycles; full-size GPU
memory viability still needs confirmation. Do not describe this infrastructure
failure as evidence against guided selection, random selection or LoRA.

The [v2 repair protocol](../experiments/instruction_boundary/transfer_protocol_v2.json)
was submitted as private kernel140453112 version1 and is now COMPLETE.
Its downloaded source matches the frozen SHA256. The data hash, eight fits,
selection/evaluation rules and runtime caps are unchanged. Preflight confirmed
no active jobs, paid scaling off and33,422.676 free GPU seconds. Three CPU checks
passed: exact FP16 output/gradient parity with recomputation, plus the full tiny
eight-fit workflow with recomputation disabled and enabled. The real comparison
now fits the GPU and completes; see the current status and outcome above.

The failed v1 run did save discovery diagnostics. Its execution manifest matches
the frozen bundle. The [summarized measurements](../experiments/instruction_boundary/discovery_diagnostic_v1.json)
show100% training classification accuracy in all72 probes, but no corresponding
advantage in the measured gradient-sharing geometry. Across projections, the
mean fraction of each support's own gradient energy preserved by shared updates
is5.83%/5.58% on the two measured training examples for guided selection, versus
6.90%/6.85% for random43 and6.52%/6.01% for random71. These are unweighted means
of per-projection ratios, not a global energy ratio or a comparison of absolute
gradient magnitude. Guided supports form a1-versus31 split in26/72 projections;
neither random support has a singleton cluster.

For orientation, under an isotropic random-gradient model, projecting32 row
directions onto two shared groups retains2/32=6.25% of energy in expectation.
The projection has rank2 in the32-dimensional row space regardless of group
sizes. This is a mathematical reference, not an empirically fitted null model or
a significance test for these correlated neural gradients. Thus6% alone is not
evidence of a broken implementation, and the singleton splits alone do not prove
harm. The observation is that this small diagnostic supplies no positive evidence
for the intended alignment advantage; trained quality remains decisive.

This is a concrete reason to question activation-derived sharing on this task,
despite excellent probe fit. It does not show that the guided update will train
worse: retained fractions normalize different supports, Adam changes geometry,
and only two examples were measured. Finish the frozen comparison before choosing
another method. If shared groups underperform, inspect grouping and actual update
transfer before concluding that identifying useful neurons is impossible.

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
