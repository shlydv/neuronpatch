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
- [ ] Add row-edit parameterization with exact freeze/locality/merge/export checks.
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
