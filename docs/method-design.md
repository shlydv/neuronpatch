# Method design and evidence ledger

Updated 2026-10-19. Goal: develop and verify a useful neuron-targeted adaptation
method that improves on relevant published baselines. Neither reproducing a paper
nor passing unit tests completes that goal. This document separates verified
mechanisms, proposed synthesis, and outstanding evidence.

## Source audit

### FiNE

[Paper](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf);
[code](https://github.com/opanhw/fine-unified-editing-code), inspected commit
`a2489cbac0dfe5a956385734065cfa041384e7d1`.
Its knowledge editor selects neurons using activation times output-vector alignment
with a predicted-token unembedding, then learns outgoing-vector updates. Code uses
maximum token NLL, prompt-distribution KL, delta norm and repetition penalties.
Safety code contains multiple stages; the first inspected training block uses NLL
plus norm, while a later block adds KL and repetition. Do not describe every stage
as the same objective.

Important discrepancy: paper Eq. 5 writes KL(edited || original); knowledge code
calls `kl_div(log_softmax(edited), softmax(original))`, computing KL(original ||
edited). The implementation's prefix-token scope also differs from preserving
responses to separate benign prompts. Our proposed preservation objective is an
explicit adaptation, not a faithful reproduction of that scope.

### NeST

[Paper, sections III–IV](https://arxiv.org/html/2602.16835v2).
Uses max-pooled gate/up projection outputs, linear probes, standardized positive
probe coefficients, and activation-profile clustering. Shared updates act on
selected gate/up output neurons (PyTorch weight rows), rather than our down-projection
input columns. Ordinary supervised training mixes harmful and benign examples.
Clustering restricts degrees of freedom; its rationale includes measured gradient
alignment within clusters. Its task is safety hardening, not document grounding.

An archival source link **does exist**:
https://anonymous.4open.science/r/nestSP27-1C20. Retrieval on this audit returned
`not_connected`; implementation has not yet been inspected. This corrects the
incomplete earlier source discovery. Reported utility is not perfectly preserved:
headline averages and model-specific drops must both be considered. Paper-supported
mechanism is not code-verified behavior.

### SafeNexus

[Paper, section 3](https://arxiv.org/html/2607.28969v1).
Scores averaged gated activations weighted by down-projection column norm, excludes
normal-prompt top candidates, and intersects modality-specific sets. Suppression
experiments compare layer-count-matched random controls. Its trainable calibrator
masks LoRA **rows in gate/up projections**, not down-projection columns. It also
studies an inference-time amplification alternative. Ablation establishes a role
in existing behavior; it does not by itself establish optimal trainability.
Official implementation remains unverified; avoid translating unspecified pooling
or masking details into claims of exact reproduction. Cross-modality intersection
is evidence for that setting, not automatically a recipe for text-format invariance.

### SpaRTA

[Paper](https://arxiv.org/html/2502.15975);
[code](https://github.com/IBM/SpaRTA), inspected commit
`e3b91c598090a306b68005ff3f9fbba1eed6c84a`.
Randomly trains individual scalar parameters with compact indexed deltas. Whole
random neuron columns are not its method. The paper's demonstrated tasks emphasize
classification, with an adapted classification head, rather than unrestricted QA.
Our comparison needs scalar-count and tuning-budget accounting, including any head
parameters, rather than calling our random-column control SpaRTA. The useful design
lesson is that locating interpretable neurons must earn its discovery cost against
strong random sparse baselines.

## What our evidence actually says

The old down-column update shifted UNKNOWN-versus-answer first-token margins
upward on every supported pilot example. Both homemade selectors over-abstained
on new families despite perfect validation accuracy. Random columns achieved a
better preservation trade-off in the follow-up; LoRA had better aggregate accuracy
than our guided methods. None of this tests the published methods themselves.

There are at least four separable choices: which behavior is detected, which
parameters can change, what objective trains them, and which contexts constrain
the update. Calling all four "neuron selection" hid these differences.

## Proposed synthesis (hypotheses, not established improvements)

1. **Discover candidate mechanisms, then test update suitability.** Candidate
   proposals can come from probes, output-aware scores or random exploration.
   Rank them by measured correction/preservation effects of small trial updates
   across calibration groups, not solely activation correlation or gradient size.
   Reuse validation folds for discovery only when labeled exploratory; retain an
   untouched confirmation set. Compare cost against random selection.
2. **Separate activation control from output content.** Support gate/up row edits
   as well as down-column edits. Hypothesis: changing activation conditions can
   improve conditional behavior without a broad output bias. This is not proven;
   gate changes can also be highly disruptive. Compare equal parameter budgets.
3. **Constrain training where preservation matters.** Add explicit reference-output
   KL on supported and unrelated examples, with well-defined token masks and
   per-example weighting. Never force the base's wrong answers on correction
   examples. Test KL direction and coefficient rather than assuming either wins.
   This can be combined with either rows, columns, sparse scalars or ordinary LoRA.
4. **Share updates only when it helps.** Clustering offers a capacity constraint,
   not a guaranteed accuracy improvement. Compare independent versus shared
   updates at both matched parameter count and matched selected-neuron coverage;
   report these as distinct experiments.
5. **Expose the trade-off.** Keep all non-dominated correction/preservation settings
   and their uncertainty; do not collapse the library into one arbitrary pass/fail
   score. A no-edit candidate remains available. Explore freely, but label the
   discovery data and confirm improvements on fresh inputs.

## Work sequence and completion evidence

- [x] Correct the intervention-axis and objective distinctions in the source audit.
- [ ] Inspect accessible NeST/SafeNexus implementations or explicitly document
      paper-only adapters and unresolved ambiguities.
- [x] Test reference-distribution loss numerics, masking, gradients and batching.
      Both KL directions are explicit; reference logits are detached. Behavioral
      effectiveness remains untested by these numerical checks.
- [x] Add row-edit parameterization with exact freeze/locality/merge/export checks.
      Row updates preserve legacy column loading; both are tested on gated Qwen.
- [ ] Compare objective and intervention axis factorially, keeping selection fixed
      first; separately test selection and clustering once objective behavior is known.
- [ ] Use real documents, counterfactual evidence pairs and hard answerable cases;
      score extraction, unsupported claims, unnecessary refusals and unrelated utility.
- [ ] Run faithful relevant baselines where technically possible, with equal search
      budgets, several seeds, and discovery/training/inference costs.
- [ ] Confirm a measured Pareto improvement on at least two task families and two
      model sizes, with fresh confirmation data and uncertainty estimates.
- [ ] Turn the successful method into a documented, reproducible user workflow;
      independently test installation, training, export and deployment.

The last two requirements define an initial defensible superiority claim in a
specified scope, not universal superiority to all papers. User's broader goal stays
active until the method, evidence and usable library support the claim. An arbitrary
universal "best" claim is not empirically supportable. No current result establishes
superiority, and no single small synthetic success will be labeled completion.

## Objective experiment update

[Controlled exploratory ablation](../experiments/objective_ablation/REPORT.md)
held supports fixed and varied response-distribution KL. It recovered supported
answers for selected columns and LoRA, with different correction trade-offs.
Therefore the objective is not a dispensable detail. This is evidence for the
proposed factorized design, not yet for a new superior method. Exposed synthetic
examples cannot serve as final confirmation. Gate/up row editing and compatible
sharing remain the next implementation questions, alongside stronger data.

## Intervention-axis experiment update

The [18-setting axis ablation](../experiments/axis_ablation/REPORT.md) found that
selected up/gate rows preserved supported answers better than the same supports
updated through down columns at the fixed learning rate. Selected up is only one
correct abstention ahead of random up. This advances an intervention candidate,
not a superiority claim. Explicit KL interacts with axis and parameterization.
Real-document corpus provenance is prepared under `experiments/real_qa`; no
model outputs on those passages have been produced. Independent confirmation,
stronger localization, shared-capacity tests and user workflow remain outstanding.

## Real-passage synthesis update

The [real-passage comparison](../experiments/real_qa/ADAPTATION_REPORT.md) tested
paired output-alignment discovery, up-row edits and reference-correct preservation
against random up rows and LoRA. Selection did not beat random on the observed
trade-off; stronger updates mostly increased refusals. Some EM gains were only
wording corrections. Preservation on 20 exact-correct training anchors did not
improve selected-row supported validation metrics at either tested rate. These
negative findings constrain the synthesis: output-aligned activation is not yet a
validated proxy for conditional update suitability. Before a new selector, inspect
training/calibration behavior of the saved fits to separate learning failure from
transfer failure and audit the preservation anchors. Confirmation remains unused.

## Learning/transfer diagnosis

[Saved-fit diagnosis](../experiments/real_qa/DIAGNOSTIC_REPORT.md) finds genuine
preservation failures on training anchors, not just unseen articles. All fits
increase refusal margins on both supported and unsupported questions, with decreased
mean within-pair separation under the first-token diagnostic. Reloaded validation
outputs match exactly. The next objective test is repeated anchor preservation
throughout training; its extra compute and established nature must be explicit.
It is a test of a concrete failure mechanism, not a new localization claim.

## Repeated-anchor preservation result

[Replay comparison](../experiments/real_qa/REPLAY_REPORT.md) improves the stronger
selected-row fit (10→15 protected training matches; 2→4 supported validation matches;
correct refusals remain 9), but leaves genuine anchor/validation refusals. It costs
roughly twice the training time and does not establish localization value. Before
another selection heuristic, a broader all-MLP LoRA reference will test whether the
model/data/objective can learn conditional support, since prior LoRA was restricted
to three layers. Any larger-budget reference must be labeled as such, not used for
an unfair efficiency or superiority claim.

## Rechecked training-scale gap

Re-reading [NeST sections V–VI](https://arxiv.org/html/2602.16835v2) highlights a
material mismatch beyond parameterization: its reported training uses a balanced
20,000-example dataset and five epochs; our real-QA development uses 128 examples.
It also checks whether activation clusters have aligned weight gradients, providing
a reason for sharing updates rather than clustering solely for compactness. These
are design lessons, not evidence that our task will reproduce its safety results.
The current all-MLP reference is a learnability diagnostic at tiny data scale;
negative results cannot establish that selective adaptation or clustering fails.
A serious next benchmark needs a learning curve over data sizes and stronger held-out
coverage before comparing selectors or claiming superiority.

The pilot's fixed layers (16, 20, 23) are also a coverage restriction, not a learned
localization result. A future equal-budget discovery comparison should search across
eligible layers and compare against layer-matched random supports. Broader LoRA can
expose a coverage problem but cannot alone show which neurons are responsible. Do
not credit neuron selection for benefits from changing layer coverage or data size.

## Broader reference and next benchmark

[All-MLP LoRA reference](../experiments/real_qa/CAPACITY_REPORT.md) fits the tiny
training corpus much better (up to 52/64 supported EM and 64/64 correct refusals)
but still damages supported validation answers. This narrows the diagnosis: the
three-layer restriction was not a complete explanation, and tiny-data transfer is
now an explicit bottleneck. Nested 128/512/2048 training sets plus expanded
article-disjoint development sets are prepared without new predictions. The next
comparison should hold model/objective fixed across data sizes before crediting a
new selector. A tested response-only vocabulary projection makes the broader local
experiment feasible without changing the causal loss.

## Common validation for the data-size curve

The expanded baseline now evaluates 2,048 training and 256 calibration/validation
examples each, reproducing all original 192 development outputs. Preservation
anchors are selected only from the corresponding training prefix (20/91/360 for
128/512/2048). Existing 128-example checkpoints have been evaluated on the same
256 validation rows as the upcoming larger fits, avoiding changing the evaluation
population between sizes. On that common set they still trade supported-answer
retention for correct refusals. The first size curve uses one epoch per size and
explicitly reports extra optimizer steps and anchor coverage; it does not isolate
data diversity from compute or claim an optimally tuned recipe at every size.

## First larger-data comparison

The [512-example comparison](../experiments/real_qa/scale_512/REPORT.md) improves
both supported exact match and correct refusals at .0003 relative to the same-rate
128-example reference on identical validation rows. At .0001 it worsens supported
answers. Interference remains: the .0003 fit preserves only 37/45 base-exact-correct
supported answers and still fails the presence/absence diagnostic. Report data,
step-count and anchor-coverage changes together. The 2048-example fits are not yet
part of the evidence; the full curve and semantic output review remain necessary.

## Semantic preservation gap

The [512-example output audit](../experiments/real_qa/scale_512/REPORT.md#semantic-audit-of-the-0003-checkpoint)
changes the next design question. Nine new exact matches contain only two clear
answer corrections, alongside wording changes, a qualifier, an ambiguity and a
question/gold mismatch. Seven formerly exact-correct answers and nine inspected
valid nonexact answers become refusals. These are selected diagnostic cases, not
an overall semantic accuracy estimate.

Our exact-match-only training anchor policy can exclude valid answers. Separately,
mean distribution KL does not guarantee preservation of the generated decision.
Neither issue is solved by finding highly active neurons. Before adding another
selector, investigate preservation coverage and decision margins on training
examples, then test whether candidate parameter updates improve missing-evidence
decisions while preserving supported decisions. This connects the useful parts of
the papers—localization, constrained updates and preservation—to a measured failure.
It does not establish that any proposed combination is novel or effective.

Keep three measurements distinct: semantic correctness, requested output format,
and appropriate refusal. Preserve published benchmark scores alongside diagnostic
annotations. Do not infer a new anchor rule from validation labels or tune it on
the untouched confirmation set. Finish the existing size curve unchanged; any
new objective needs a same-data, same-budget control before attributing gains to
neuron selection. Reproduction is useful only where it resolves an implementation
uncertainty; it is not a substitute for this synthesis objective.
