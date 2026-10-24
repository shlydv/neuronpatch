# Current method decision — 24 October 2026

**Goal:** a robust sparse-adaptation library with a verified practical advantage
in quality, preservation and cost against strong baselines. **Not achieved.**
The library can train, save, reload and merge compact patches. No neuron-selection
recipe has yet earned recommendation as a default or a superiority claim.

## One active experiment

[Matched-authority discovery](https://www.kaggle.com/code/shlydv/neuronpatch-boundary-authority-v1-20261024),
private kernel140580989 version1, is running. Its two fits and new-candidate
answer generation finished; saved-comparator replay has started. This is a
progress snapshot, not a quality result or full completion. Paid scaling is off;
provider/worker/internal limits are10800/10200/9900 seconds. No other GPU job is
part of this plan.

The [frozen protocol](../experiments/instruction_boundary/authority_protocol_v1.json)
changes discovery prompts while preserving training targets, budgets, rates and
optimizer. Both discovery classes contain the same redirect text, placed inside
untrusted JSON versus outside it as a legitimate user instruction. Grouping is
recomputed from those profiles, so this evaluates the discovery recipe, not
ranking alone. The32 training documents remain unchanged. Fresh confirmation
uses48 documents, three attack families and legitimate-instruction controls.
Four comparator adapters are reused without new fitting. The full plan contains
3216 teacher/confirmation generations and two new fits.

Completion requires downloaded full-comparison artifacts, independent integrity
verification and complete answer review. A candidate-only completion file does
not suffice. The [source-only review](../experiments/instruction_boundary/authority_source_review_v1.json)
is frozen before outputs: correct references6/9/45, accept valid alternatives46,
exclude document14 QA for every condition, retain its controls, and apply the
predeclared document43 sensitivity. Review anonymously before joining condition
identities. Judge actual recovered answers, not just disappearance of attack text.

## Strongest completed evidence

[Boundary-transfer v2](../experiments/instruction_boundary/complete_v2/REPORT.md)
completed eight fits and1728 generations with verified artifacts. Qwen3-4B-
Instruct-2507 answers27 clean documents correctly; the table below reports recovery
on those same27 documents. Quality judgments were made by the assistant, not an
independent human reviewer.

| Method | Familiar redirect | Unseen claimed authority | Clean regressions |
|---|---:|---:|---:|
| LoRA |25/27|21/27|0|
| Guided grouped patch |24/27|0/27|2|
| Random grouped, seed43 |25/27|19/27|0|
| Random grouped, seed71 |25/27|4/27|3|

All192 legitimate direct/quoted controls pass for every condition. This is limited
control coverage, not broad capability preservation. Grouped patches train368640
scalars versus LoRA2064384, but fit cycles were about364 versus349 seconds:
**fewer parameters did not establish a speed advantage.** Guided selection fails
the frozen quality, transfer and preservation criteria.

Earlier [64-label](../experiments/relation_learning/complete_v2/REPORT.md) and
[eight-label](../experiments/relation_learning/complete_low_data_v1/REPORT.md)
relation tasks also showed learning without a selected-neuron advantage. The
[math baseline](../experiments/relation_learning/math_baseline_complete_v1/REPORT.md)
had insufficient completed errors and was retired without adaptation.

## What the diagnostics do and do not establish

- The new [document-held-out recognition check](../experiments/instruction_boundary/complete_v2/probe_crossvalidation.json)
  selects32 features within each training fold and refits its probe. Accuracy is
  4603/4608(99.89%) across72 projections on the same64 cases. These are repeated
  measurements, not4608 independent documents. It supports recognition across
  documents within one template, not transfer to new attacks or useful editing.
- Guided groups split31/1 in26 of72 projections; random groups never do. This
  finding was already recorded before takeover. The new
  [saved-adapter audit](../experiments/instruction_boundary/complete_v2/adapter_geometry.json)
  reproduces it and adds effective update norms. Imbalance alone is not an
  explanation: the two random patches have similar occupancies but different
  outcomes. Do not prescribe balanced clusters without evidence.
- Two-example gradient-sharing diagnostics gave no positive alignment evidence.
  A retention fraction near6% is not intrinsically broken: an isotropic32-row,
  two-group reference retains6.25%. The sample is too small for a general claim.
- Guided/random comparisons change support and induced sharing together. LoRA
  additionally edits attention and MLP output projections, while grouped patches
  edit gate/up only. The practical comparison is valid; the causal diagnosis
  needs these differences separated.

The [selection decision record](selection-decision-2026-10-24.md) contains the
measurements, interpretation limits and outcome-dependent next decisions.

## Decision after the active run

Honor the frozen success criteria, then inspect failure modes. If the new recipe
beats random supports and approaches LoRA while preserving answers, replicate
across seeds and tasks before recommending it. If it improves only the old guided
patch, that is discovery-recipe improvement, not a competitive method. If it
still fails, do not launch another arbitrary pooling, threshold or formatting
sweep. Identify whether support, sharing or training coverage limits transfer;
a new experiment must discriminate those explanations.

We need selection to predict useful finite training changes on separate transfer
calibration examples, not merely large gradients or accurate input recognition.
This remains an unvalidated direction with substantial prior art, not a new
algorithm already established by our work. Never tune on confirmation outputs.

## Paper provenance and completion requirements

[Research provenance](research.md) and [prior design audits](method-design.md)
distinguish actual reproductions from hypotheses. FiNE has bounded upstream
workflow checks; NeST's shared-row operation has CPU parity, not a full safety
replication; SafeNexus official code remains unverified. Our fixed two-group,
32-row adaptation is not the full NeST selection/clustering method.

A successful result must improve held-out quality/preservation at lower label,
parameter, storage or measured compute cost against tuned LoRA and random controls.
It still needs additional seeds, a second task, meaningful unrelated-task checks,
a model-size check, full discovery/training costs and verified exported behavior.
Working primitives, toy tests, fewer scalars or paper-inspired terminology do not
fulfill the goal.

The [archived accumulated snapshot](method-history-2026-10-24-pre-takeover.md)
and [experiment history](experiment-history.md) preserve earlier evidence and
source links. Historical candidate plans are not instructions to launch jobs.
Update this page by replacing stale state, not appending another current status.
