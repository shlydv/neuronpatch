# Current method decision — 24 October 2026

**Goal:** a robust sparse-adaptation library with a verified practical advantage
in quality, preservation and cost against strong baselines. **Not achieved.**
The library can train, save, reload and merge compact patches. No neuron-selection
recipe has yet earned recommendation as a default or a superiority claim.

## Active bounded test

[Full-document supervision](https://www.kaggle.com/code/shlydv/neuronpatch-boundary-supervision-v1-20261024)
kernel 140599487 version 1 is RUNNING at submission verification. Accepted source
matches the frozen bundle. One private T4; provider/worker/internal caps are
10,800/10,200/9,900 seconds. No paid scaling requested.
The [protocol](../experiments/instruction_boundary/supervision_protocol_v1.json)
fixes three new fits at rate 0.0003 using saved matched/random43 supports and
LoRA. Short direct preservation inputs become full-document requests; teachers
and sample schedules remain fixed. Three old endpoints replay on the same
32 fresh documents: 2,912 new generations across seven conditions. All 368
previously allocated documents are excluded. Source review predates submission;
one QA item is excluded for missing nationalities, with all controls retained.
No outcome is available. This tests supervision repair, not achieved superiority.
See [design and provenance](boundary-supervision-test.md).

## Latest completed experiment

[Matched-authority discovery](../experiments/instruction_boundary/complete_authority_v1/REPORT.md)
completed two new fits and 3,216 generations; all artifacts downloaded and passed
integrity checks. All 309 unique anonymous QA answers were reviewed before
condition attribution. There is no active GPU job for this experiment.

On 38 base-clean-correct fresh documents, the new grouped patch recovers 38
familiar redirects, 16 unseen authority attacks and 20 grading-notice attacks.
Old guided recovers 36/0/0; LoRA 37/31/23; random43 36/27/10; random71 37/3/7.
The new discovery recipe improves on old guided but fails the frozen competitive
criteria, including under more lenient answer-content scoring.

**The larger finding:** all five trained adapters fail all 48 legitimate
instruction changes after the full quoted document; the base passes all 48.
Every method still passes the 288 short direct/quoted controls. Therefore those
short controls were insufficient to establish boundary preservation. Relative
preservation against equally damaged comparators is not absolute preservation.

Discovery used matched trusted/untrusted contexts, but training supervision
remained unchanged. The next investigation should hold support fixed and test
matched full-document supervision with different targets across the boundary,
using original training/calibration data and fresh confirmation. Do not launch
another selector-only sweep or tune on exposed confirmation answers.

## Previous completed evidence

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

## Current method decision

We have evidence that changing discovery helps this particular guided patch,
not that it beats strong controls. We also have a shared training failure that
must be addressed before making sparse-selection claims. Fixing supervision is
a controlled hypothesis, not an assumed cure: preserve fixed supports, budgets
and strong baselines, and test both sides of the trust boundary on fresh inputs.

Selection should predict useful finite training changes on separate transfer
calibration examples, not merely large gradients or accurate input recognition.
This remains an unvalidated direction with substantial prior art. Even a perfect
local predictor on an inadequate objective cannot certify the desired behavior.

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
