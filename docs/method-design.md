# Method design and evidence ledger

Updated 2026-10-20. Goal: develop and verify a useful neuron-targeted adaptation
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

## Shared-row primitive prepared

`neuronpatch.grouped.GroupedRowPatch` implements the shared additive row update
from [NeST equation 13](https://arxiv.org/html/2602.16835v2), given explicit indices
and groups. It does not implement neuron discovery or automatic clustering.
The archival implementation remains inaccessible on recheck, so this is a
paper-based implementation, not code-verified reproduction.

Tests verify dense-forward equivalence, summed group gradients, input gradients,
untouched base weights, merge locality and equivalence to independent rows when
all groups are singletons. Group size changes gradient accumulation; equal
trainable-parameter count does not imply equal optimization behavior. The module
is experimental; high-level checkpoint support was added subsequently (below).
No model-quality advantage is established, and it is not part of the coverage
experiment. It prepares an independent sharing ablation once preservation is
better understood; passing primitive tests cannot establish the combined method.

## Completed data-size curve

The [2048 comparison](../experiments/real_qa/scale_2048/REPORT.md) completes both
rates at all three sizes. Both largest fits reach 50/128 supported EM and 94/96
correct missing refusals, versus base 45/34. Unnecessary refusals remain 43/48
versus 13, and only 38/33 of 45 base-exact answers survive. Broader data/compute
helps some trade-offs but does not establish reliable preservation. Protected
training anchors also break. Semantic review and article-bootstrap intervals
prevent interpreting five extra matches as five unambiguous new correct answers.

The next run is the prepared coverage-only comparison. It evaluates one measured
weakness while retaining the same objective, model and budget. Sharing and
localization remain independent hypotheses requiring later matched comparisons;
their numerical implementation alone does not satisfy the synthesis goal.

## Coverage outcome and decoder contribution

The [coverage comparison](../experiments/real_qa/coverage_512/REPORT.md) does not
establish improvement: new-anchor semantic retention changes 8→8 and 12→13/24,
while stronger-rate supported validation EM declines 46→39/128. Protection
coverage is insufficient under the original decoding settings.

The subsequent raw-logit audit reveals an overlooked generation setting:
repetition_penalty=1.1 remains active with greedy decoding. On five selected
training examples, it flips the correct raw first-token preference to UNKNOWN;
neutralizing it recovers valid answers. Eleven default outputs reproduce exactly.
This is a demonstrated decoding contribution, not proof of a general cure. A
paired neutral-decoding evaluation of base and all four checkpoints is now needed
before introducing a new preservation objective. Keep raw-logit preservation and
actual decoded behavior distinct when evaluating every proposed method.

## Decoder comparison completed

The [five-model decoder comparison](../experiments/real_qa/decoding_512/REPORT.md)
finds a genuine but limited effect: neutral repetition penalty reduces unnecessary
refusals by 4–6/128, while reducing correct missing refusals by 3–6/128. Expanded
coverage remains worse on supported answers at the stronger rate under either
decoder. Do not promote a decoder correction as a new adaptation method. Record
effective generation settings explicitly and investigate the actual response-token
preservation loss before assuming a different selector will fix interference.

## Token-loss diagnosis

The [token-level audit](../experiments/real_qa/coverage_512/TOKEN_LOSS_REPORT.md)
matches the actual loss numerically and finds the first token is the largest KL
in 135/139 refused anchor/checkpoint observations. Its error is typically diluted
four to six times by response averaging. This motivates a max-token preservation
ablation, with stronger mean KL as a required control. The
[protocol](../experiments/real_qa/coverage_512/TOKEN_LOSS_PROTOCOL.md) isolates that
objective change under common neutral decoding before testing shared or selected
neuron updates. This is a measured synthesis direction, not an established new
method or an implication that every first-token change is semantically harmful.

## Shared-update artifact support

Explicit output-row groups now work through NeuronPatch's attach/save/load/merge
API. Format 3 stores group assignments and one delta per group, with the same
strict base fingerprint contract. Tests cover compact parameter count, round-trip
output equivalence, shared merge deltas, unchanged nonselected rows, legacy
compatibility, malformed grouping rejection before mutation and nested-patch
rejection. This closes a usability gap for a later sharing experiment. It neither
selects groups automatically nor establishes that sharing improves preservation.
The running objective comparison still uses ordinary all-MLP LoRA.

## FiNE reference claim audit (2026-10-19)

The [upstream reference audit](../experiments/fine_reference/CLAIM_AUDIT.md)
corrects a too-broad expectation: the GPT-J localization ablation improves edit
success and subject-alias transfer while random selection scores higher on both
reported locality measures. It does not establish a universal data-efficiency
claim. Our target remains a useful, measured trade-off, not all-metric dominance.
The reference harness calls unchanged upstream editing/scoring functions, with
pinned source/model/data. CPU model and metric checks passed at this stage. The
initial GPU-quota diagnosis was wrong: subsequent GPU allocation and the smoke
result below disproved it. The original driver's unused
`--random` argument is bypassed by setting its actual hyperparameter explicitly.
The first 16 cases are only an execution/diagnostic subset, with sparse coverage
of some evaluation categories; do not label that as a full table reproduction.

## Working upstream FiNE smoke result (2026-10-20)

The [two-case GPU reference](../experiments/fine_reference/smoke_v4/REPORT.md)
completed with unchanged upstream editor/scoring functions. Selected neurons
produce both requested counterfactual edits and all10 alias answers; random
selection produces neither original target nor any alias target. Selected edits
alter unrelated predictions, and both generated relation rephrasings still fail.
One failed generated rephrasing scores75% teacher-forced accuracy. This establishes
execution and an observed tiny-sample selection effect, not full replication or
library superiority. Source/model/data are pinned, all restore checks pass, and
actual outputs are saved. Python compatibility and CUDA0 placement resolved
execution issues. The same first16 cases are now running as expanded exploratory
evidence before adapting the reference method into NeuronPatch.

## Portable contribution selector (2026-10-20)

`contribution_indices` now exposes the FiNE all-eligible-layer, per-predicted-token
score in the library. Its selected indices match the unchanged upstream scoring
routine on18 CPU fixtures/72 token positions, with CUDA calls redirected only
inside that local test. This establishes numerical CPU selection parity, not GPU
or behavioral equivalence of a complete library editor. All63 library tests pass
with2 MPS cases skipped in the sandbox. The running first16-case benchmark still
uses unchanged upstream code; library changes do not alter that experiment.

Repeated selections across target positions are deliberately retained. Upstream
trains one delta per occurrence and penalizes their joint norm, so deduplicating
before training is an optimization change. The existing unique-column patch API
must not be advertised as an exact training replacement until occurrence and
mixed-precision semantics have been verified. This is a concrete remaining bridge
between the working editor and our library, alongside actual preservation and
rephrasing transfer. NeST's paper-linked artifact still returns HTTP403 on direct
retrieval; SafeNexus source is still unverified. No reproduction claim for either.

## Occurrence training bridge verified locally (2026-10-20)

`OccurrenceColumnEdit` retains one global trainable row per selected occurrence,
including duplicates, with explicit delta precision and the reference joint norm.
Four Adam steps match unchanged upstream `execute_fine` exactly on a tiny CPU
GPT-J with256 occurrences/128 unique features, for FP32-base/BF16-delta and
FP16-base/BF16-delta configurations. Sequential merged weights also match exactly;
base weights remain unchanged and hooks are removed. The check uses all eligible
features at two target positions to guarantee duplicates; it is a numerical fixture,
not an editing quality demonstration. Both source and test scopes are explicit.

The primitive is experimental and does not yet provide compact checkpoint support,
a complete published objective/stopping workflow, or real-GPU behavioral parity.
It is distinct from the legacy unique-column API. The fixed16-case run remains
unchanged and active; these integration checks do not establish any method advantage.
Library tests now total67 passed/2 skipped in this environment.

## Reference objective integrated (2026-10-20)

The library now implements the FiNE-K code objective with explicit causal scopes:
worst-token target NLL, prompt-prefix KL(reference || edited), post-target
repetition and global occurrence norm. Analytical tests check coefficients,
KL direction, target positions, worst-token/tie gradients and detached references.
The integrated check uses library localization, this loss and occurrence updates;
all supports, four-step learned BF16 deltas and sequential merged weights exactly
match upstream on the two tiny-model precision configurations. All75 tests pass,
with2 hardware cases skipped. This validates a faithful numerical training path,
not a new method or full-scale effectiveness. Compact occurrence artifact support
and a complete real-GPU library workflow are still missing. The unchanged16-case
reference run is progressing; no new behavioral experiment has been launched.

## Compact occurrence artifacts (2026-10-20)

OccurrenceColumnEdit now saves only its ordered targets, global delta and metadata
against an exact base fingerprint. Repeated indices and BF16-update/FP16-base
precision survive reload. Malformed shapes/dtypes/values, invalid target metadata
and base mismatches fail before mutation; saving rejects changed base weights.
The standalone edit artifact does not change existing NeuronPatch formats and
excludes optimizer state. The integrated CPU source check now round-trips learned
deltas and outputs before verifying exact sequential merged weights. All84 tests
pass,2 skipped. Real-GPU library parity and an end-to-end supported training
workflow remain outstanding; no new behavioral result is claimed. The unchanged
16-case upstream experiment continues, currently past the first four paired fits.

## Complete ordinary-HF workflow (2026-10-20)

The experimental `fit_gptj_edit` now handles the reference's token alignment,
activation capture, contribution/random selection, occurrence training, objective
and early stopping on an ordinary Hugging Face GPT-J. Exact CPU parity with the
unchanged source holds for FP32/FP16 bases, contribution/random selection,
explicit target EOS and initial zero-update stopping. Compact reload and
sequential merged weights remain exact in these fixtures. All96 tests pass,
with2 MPS cases skipped. These are numerical checks, not evidence of full-size
GPU parity or improvement. The running reference remains independent; ten paired
cases have completed with restoration checks, and six remain at this observation.

The implementation discrepancy is now substantially narrower: our earlier QA
heuristic was not their selector, our task/model/objective differed, and duplicate
occurrences, precision, joint regularization and evaluation scopes matter. This
does not establish which difference caused the earlier negative QA result. The
next behavioral decision must use the expanded reference's generated outputs and
locality failures. First validate this library workflow at 6B scale; then isolate
one improvement with matched controls and genuinely fresh confirmation. Passing
more plumbing checks alone does not meet the project goal.

The ordinary-HF integration also runs in the installed current Transformers
runtime without model downloads. A separate
[full-size parity protocol](../experiments/fine_reference/GPU_PARITY_PROTOCOL.md)
and script compare both editors under one isolated reference-compatible runtime.
They are prepared, not yet executed; runtime differences from the first16
upstream benchmark are explicit.

Source availability was rechecked against the current NeST paper and SafeNexus
author page. NeST's linked archive still could not be retrieved. The SafeNexus
author page's Code link still resolves to a different cross-architecture merging
project. Neither is treated as a verified source implementation.

## Preservation requires a working baseline (2026-10-20)

The new free-generation audit reproduces the smoke edit/alias/rephrase counts
but finds zero gold-prefix matches on all11 unrelated probes before editing.
Manual inspection finds hallucinated or non-answering continuations, and at least
one awkward relation prompt. Thus the smoke's prediction-agreement scores cannot
establish retention of useful answers; conditional generated-answer retention has
a zero denominator. This is an evaluation limitation, not evidence that no
interference occurred. The expanded run must expose baseline quality, generated
answer changes, and upstream teacher-forced metrics separately. Later method
comparisons need enough already-correct preservation examples and actual output
review, rather than optimizing stability of an already-poor baseline.

## Next selection control: separate layer choice from neuron choice

The source uniform-random control samples layers and features independently. It
does not match the guided support's layer distribution or repeated occurrences.
These are useful controls to add before attributing the entire gap to precise
neuron identity. A within-layer random bijection of selected feature IDs preserves
the layer sequence, occurrence count and multiplicity pattern while replacing
neuron identity. Compare several frozen seeds against the original and uniform
random supports with the same objective, budget and evaluator.

Duplicate occurrences are more than bookkeeping. In exact arithmetic, identical
input columns, zero initialization, one joint L2 norm and identical Adam state
give identical updates to repeated rows. With multiplicity c and per-row vector d,
the effective column update is e=c*d, while its squared contribution to the norm
is c*||d||^2=||e||^2/c. Effective update scaling and regularization therefore differ
from deduplicating supports. BF16 accumulation and sequential merge rounding can
break exact compressed equivalence. This is a mathematical observation about the
code, not evidence that multiplicity explains its performance. Case0's successful
one-token smoke edit has no duplicates, so duplication cannot explain that result.
The stronger control should establish whether fine neuron identity earns its cost;
it does not itself constitute an improved adaptation method.

## Expanded reference completed and next candidate (2026-10-20)

The [16-case upstream result](../experiments/fine_reference/reference16_v1/REPORT.md)
is complete: selected generates 15/16 original target prefixes, 5/16 rewordings
and 52/94 aliases; uniform random gets zero on those three categories. Among
18 unrelated baseline reference-prefix matches, selected retains 15 and random
17. Full-word boundaries matter: “Johnny Reineke” does not equal “Johnny Reine”
even when all target tokens score correctly. Six failed generated rewordings
receive at least 75% teacher-forced accuracy. The original metric also omits the
EOS used during editing/generation, so both conditioning differences must remain
explicit; they have not been isolated causally. All32 restore checks pass and the
two smoke cases reproduce exactly.

This confirms a working published recipe, and demonstrates limits worth fixing.
It does not establish that exact feature identity beats layer/multiplicity-matched
random supports, nor that this small exploratory sample establishes broad utility.
The library's two-case full-size port check is now RUNNING separately using
torch2.4.1 for both implementations. Its source revision and hashes are recorded.

The new [source/math audit](../experiments/fine_reference/SPARSE_PROJECTION_NOTE.md)
identifies an interaction between sparse masks and preservation projections.
NMKE's post-mask operation can leave the protected null space. In16 FP64 random
fixtures, the post-mask update changes protected outputs; a joint sparse/protected
solve preserves them while fitting the requested residual, at a larger update
norm. A support with no free direction correctly cannot implement the edit.
These are matrix results, not language-model improvements.

An axis-aware candidate follows: for incoming-neuron row updates, output-row
sparsity and right-side input preservation act on different dimensions and are
compatible. NeST-style row sharing is compatible as well. First check actual
activation rank and available directions, then test the candidate against
unconstrained rows, matched random supports and appropriate LoRA/protection
baselines with identical data and budgets. Existing AlphaEdit/NMKE and Null-Basis
LoRA must inform those controls; merely adding a projector is not novel.

The quality goal remains generated transfer and retention of already-working
behavior on fresh inputs. Exposed KnowEdit cases are development data; current
matrix checks and the library port cannot serve as that confirmation. Do not add
the new constraint to the public API or call the goal complete on this evidence.

## Full-size port passed; protected-row prototype built (2026-10-20)

The [6B library comparison](../experiments/fine_reference/library_parity_v1/REPORT.md)
completed. All four conditions exactly match the original ordered supports,
learned BF16 deltas, merged selected columns and 18 fixed merged generations.
Compact artifacts were independently rechecked against upstream deltas after
download; all four are exact, and base fingerprints agree. Eight restoration
checks pass. One live-hook continuation differs from its merged version,
confirming the documented mixed-precision limit. The original and new runtime
also agree on the18 checked baseline/merged outputs; this is not a cross-version
guarantee. Raw version1 artifact metadata copied initial random defaults; the
future runner now records effective selection metadata without changing training.

The [new prototype](../experiments/protected_rows/README.md) implements selected
incoming rows with a fixed protected input span, checks remaining capacity, and
exports effective deltas through the ordinary patch format. In a tiny nonlinear
FP64 network, it learns a representable change while keeping calibration anchors
within1.12e-16; reload and merge match within1e-12 and base weights stay unchanged.
This validates the proposed mechanism and deployment path numerically. It does
not establish improved LLM answers. The next safe action is to freeze and run the
bounded actual-model comparison described there, including unconstrained rows,
matched random controls and LoRA, then inspect held-out behavior.

## First actual-model comparison frozen (2026-10-20)

The [pilot protocol](../experiments/protected_rows/PILOT1_PROTOCOL.md) fixes a
pinned Qwen3-4B-Instruct-2507 checkpoint, mutually disjoint article partitions,
new paragraph contexts, generated-answer scoring and eight fits. All methods
receive the same 160 adaptation labels, including calibration labels used for
selection. Four actual baseline-correct calibration trajectories provide both
the protected input spans and equal preservation replay. Held-out utility does
not select learning rates. Official SQuAD development data remains untouched.

The selected row IDs are identical in the protected/unprotected comparison;
protected random rows match the layer/count budget. LoRA deliberately has a
stronger parameter budget and two rates. We report allocated and effective row
dimensions, selection cost, training cost, merge precision, full generated
outputs and restored-base fingerprints. Success requires improved held-out
correction with at least 95% retention of baseline-correct supported and utility
answers, followed by a fresh confirmation experiment. The local tiny-Qwen3
preflight passes all four training/export/restoration paths; model quality has
not yet been measured. No superiority claim follows from this preparation.

## Protection-matched control prepared (2026-10-20)

A later improvement claim must separate input protection from neuron selection.
The [null-basis low-rank control](../experiments/protected_rows/null_lora.py)
allows all output rows through the same fixed permitted input space. It is an
independent paper-based implementation of NB-LoRA's parameterization, using our
numerical span rather than the paper's 99%-energy cutoff. Tiny nonlinear and
ordinary-Qwen3 integration checks pass, including frozen original weights,
export, merge and restoration. It has not run on the actual checkpoint.

The eight-fit pilot remains frozen. Its ordinary-LoRA comparison can motivate
further work but cannot by itself show an advantage over null-basis LoRA.
If a protected row setting improves real held-out answers, the next comparison
needs the protection-matched control, matched target modules, explicit parameter
budgets, equal replay and tuning, and fresh data. This is required to determine
whether neuron sparsity/selection adds value beyond an existing protection method.

## Eight-fit actual-model pilot completed (2026-10-20)

The [frozen Qwen3-4B pilot](../experiments/protected_rows/colab_v1/REPORT.md)
completed on a verified Colab T4 after Kaggle remained queued. The queued copy was
cancelled before executing the byte-identical original payload on Colab. Model,
runtime packages, protocol and data hashes are preserved. All eight fits execute
160 steps without scaler skips; all 79 output files were downloaded and verified.
Saved row supports, shapes, fingerprints and LoRA factors pass the artifact audit.

No setting meets the correction plus 95% preservation rule. At the same missing
answer score (53/64), protected selected rows retain 29/38 baseline exact-correct
supported answers, versus 25/38 for weaker unprotected selected rows. This is a
partial exploratory signal, not established superiority. Weak protection retains
35/38 but gains only 2/64 missing answers. Strong LoRA gets 56/64 missing answers
and retains 27/38. Protected random rows do not improve aggregate accuracy.

Finite anchors are insufficient: every family, including ordinary LoRA and
unprotected rows, preserves the four replayed anchor trajectories exactly, yet
new correct answers break. The protected span has rank 753/2560 in each target
layer. Actual answer inspection exposes both genuine unnecessary refusals and
exact-match artifacts. Of 26 supported baseline exact-match misses, an assistant
audit judges 19 to have correct content; this is not independent human labeling.
Some missing labels are explicitly contradicted by their passages. Frozen primary
scoring remains unchanged. The stronger protected setting repairs no reviewed
genuine supported-content failure and refuses six of those 19 correct-content
answers; cosmetic exact-match gains are not evidence of better reasoning.

Before another training experiment, the [dose diagnostic](../experiments/protected_rows/dose_protocol.json)
is frozen and running on the cached model: scale the unprotected patch to match
calibration refusal-log-odds change, and compare a constant first-token UNKNOWN
bias. This separates conditional behavior from a general refusal threshold.
It also cold-loads both high-dose exported row patches and requires exact
reproduction of all 320 validation/utility continuations. It uses already exposed
data and cannot serve as fresh confirmation. No new method is promoted yet.

## Refusal-strength diagnostic completed (2026-10-20)

The [520-second saved-weight diagnostic](../experiments/protected_rows/dose_v1/REPORT.md)
completed and was archived before releasing the temporary GPU. Public cold-load
and FP16 merge reproduce all 320 saved validation/utility trajectories exactly;
the final full base fingerprint matches. The unprotected scale 0.47603 matches
protected calibration UNKNOWN log-odds shift within 0.60%, without validation
selecting the dose.

Protected: missing 53/64, old-correct supported retained 29/38. Matched unprotected:
50/64 and 32/38. A first-token-only constant UNKNOWN bias on the base gets 48/64
and 35/38. These points trade correction against preservation; no setting passes
the original criteria. The simple bias captures part of the gains without learning
weights, so it is a required baseline for future claims. It does not reproduce
answer reformatting or every failure of the trained patches.

Secondary first-token missing-label AUROC decreases from base 0.9080 to protected
0.8938 and matched unprotected 0.8970; constant bias leaves ordering unchanged.
No significance or clean-label claim is made. The measured separation does not
improve, despite more refusals. The assistant-reviewed subset exposes both true
missing-answer fixes and destruction of correct-content answers omitted by
strict exact-match retention. This is a behavioral failure mechanism rather than
an unexplained export implementation mismatch.

Stop tuning this recipe on exposed data. The next hypothesis is a correction
objective and neuron-selection signal that reward differential evidence response
on checked contrasting contexts, with a constant refusal shift unable to satisfy
the separation objective. Keep calibrated threshold-only and random/low-rank
controls. This is a direction to build and falsify, not a new proven algorithm.
Source parity and finite-anchor invariance remain useful parts of the library;
neither is sufficient evidence of a generally useful behavior adaptation method.

## Conditional objective and reviewed controls built (2026-10-20)

The [evidence-pair experiment](../experiments/evidence_pairs/README.md) now has
a tested pairwise first-token objective. Within-pair UNKNOWN-versus-canonical-
answer logit differences cancel arbitrary shared vocabulary biases, including
question-specific ones. A reference-center penalty prevents such a bias from
lowering the total objective starting at the base. This is an established ranking
construction applied to a measured failure, not a novel ranking algorithm or
proof of better answers. Conditional context-only shortcuts remain possible.

Both variants remove a source unit, with length-matched unrelated removal as the
supported control. Source-only assistant curation selects 64 training pairs from
100 reviewed candidates, rejecting surviving aliases/inferences, false premises,
ambiguous gold and malformed questions. Each selected pair also has a checked
retained-fact question answerable under both edits. Thus 64 articles yield 128
distinct questions and 256 context labels, not merely 64 supervision examples.
All cache IDs and real-tokenizer lengths pass the training data check. These
assistant labels are not independent human gold.

This controlled atomic-fact development diagnostic does not replace broad QA,
untouched transfer or the full library objective. Calibration/validation curation
is still incomplete, and finalization refuses partial signoff. No GPU run has
been launched on the new data. Next: finish source-only review, freeze data and
a bounded objective/control comparison, then check learnability and transfer.
Keep calibrated no-training thresholds, equal data, supported/invariant retention,
random/low-rank controls and actual generated answers explicit.
