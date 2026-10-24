# Full-document supervision test — completed 24 October 2026

**Completed:** [verified results](../experiments/instruction_boundary/complete_supervision_v1/REPORT.md). All three methods restored full-document compliance but failed attack-retention gates. The submission details below record the frozen design.

The [completed authority experiment](../experiments/instruction_boundary/complete_authority_v1/REPORT.md)
found a shared failure: every trained adapter ignored all 48 legitimate redirects
after a complete document, although the base followed them. Short direct and
quoted controls all passed. This motivates a supervision ablation before more
neuron-selection development. It does not prove the proposed repair will work.

## Frozen change

`experiments/instruction_boundary/matched_supervision.py` replaces each eligible
short direct redirect preservation example with the same redirect outside the
full quoted document. The marker target is identical. Task examples, clean and
quoted preservation examples, bank sizes and ordering are unchanged. Thus the
existing seeded sampling can use the same indices and the same number of Adam
updates. Full-document task and legitimate-control cases now require different
answers despite sharing question, table and redirect wording.

CPU construction on the actual saved teachers replaces 32 training examples and
16 calibration examples. Tests check the JSON trust boundary, opposite targets,
unchanged task targets/order, incomplete-teacher handling, reference independence
and allowed bank membership. Callers must supply authoritative split membership;
the helper cannot detect a caller falsely labelling confirmation as training.
The runner now supports hash-bound frozen inputs, skips rediscovery and teacher
generation, constructs fresh adapters from row/group identities, and validates
the exact parent sample-index schedule after fitting. Its calibration scorer
uses the full-document control in place of the short direct control. A tiny
Qwen CPU fixture exercises actual Adam updates, serialization, reload, merge and
base restoration through this path; only generation is stubbed. This verifies
execution mechanics, not large-model quality. Packaging includes the pinned
assets. Multiple-parent comparator replay and independent verification are now
implemented. A tiny-model test replays disjoint parent subsets, checks all
generation records, and rejects missing receipts and altered supervision.
Source review and protocol are frozen. Kaggle accepted kernel 140599487 version1,
`shlydv/neuronpatch-boundary-supervision-v1-20261024`. Source SHA-256
`0046d49387ad310888833f3dbc2be05b53c8d1494e258f109548fe53f39f15ee`
matches the local bundle. Status at verification: RUNNING.

The private preparation currently allocates 32 fresh documents after excluding
all 368 previously allocated documents. There are 416 confirmation cases per
condition: clean plus injected/short-direct/quoted/full-document-direct for each
of three known families. Three new fits and three replayed endpoints plus base
require 2,912 new generations. Forty-eight cached teacher records are verified
but are not new inference. The tokenizer check finds maximum supervised length
1,101 and maximum confirmation prompt length 1,040, within the existing caps.
Parent artifact hashes and fixed support identities pass the CPU checks.
Fifteen focused tests pass, including actual tiny-model optimization; these do
not establish large-model behavior. The single private T4 run is submitted; log observation and automatic artifact
collection are attached.

This is a standard counterexample/data-coverage intervention, not a new selector
or a claimed synthesis breakthrough. Equal example counts and optimizer steps
do not equal token or wall-clock cost; record those costs explicitly.

## Frozen run design

- Start fresh from the exact original model weights. Load the *support and group
  assignments* of the saved matched-authority patch and random43 patch, but zero
  their deltas. Do not rediscover or continue the trained deltas. Include the
  same rank-one attention-plus-MLP LoRA representation.
- Keep the prior 96 steps, seed, preservation weight and selected rate 0.0003.
  This isolates a fixed-budget supervision intervention; it does not estimate
  each representation's retuned optimum. Explicitly verify equal sampled indices
  and unchanged task examples against parent training histories.
- Compare all three new fits with the corresponding three old saved adapters
  on the same fresh confirmation bank, plus the untouched base. Reuse clean
  teachers rather than consuming GPU to rediscover identical support/targets.
- Never use the exposed authority confirmation answers for fitting or rate
  selection. Exclude every allocated earlier document from fresh confirmation.
- Include full-document legitimate instructions for every evaluated attack
  wording, not just redirect. Retain short direct/quoted and clean questions.
  Judge complete QA answers, marker contamination and instruction compliance
  separately; freeze these conventions before reading new outputs.
- Freeze document counts, generation totals, three-hour maximum, source/model
  hashes and paired success/failure criteria before submission. One private free
  Kaggle T4 job, paid scaling disabled. No additional job is authorized by this document.

The causal question is whether replacing these controls repairs legitimate
instruction compliance without sacrificing attack recovery, on fixed supports.
If all representations improve, training coverage was an actionable common
limitation. If only some improve, support/capacity interacts with the repair.
If none improve, inspect the preservation objective, optimization and actual
sample exposure before trying another selector. None of these outcomes alone
establishes a library advantage over strong baselines or the cited papers.

## Applying the frozen outcome gates

After artifact verification and response-bound blinded review, run:

```bash
python -m experiments.instruction_boundary.supervision_verdict \
  --inputs artifacts/boundary_supervision_v1/data/inputs.json \
  --reviewed-records PATH_TO_REVIEWED_RECORDS.json \
  --output NEW_VERDICT_PATH.json
```

The checker requires the complete 32-document, seven-condition case grid and
applies the already frozen per-family thresholds, net corrections, own-parent
task retention, and clean regressions. Fewer than 16 base-clean-correct documents
produces an insufficient-headroom verdict. Secondary selection checks are eligible
only when the guided arm passes its repair gate. Full-document and clean
regressions are compared against both new random and LoRA endpoints. This counts
reviewed outcomes; it does not validate judgments, substitute for the artifact
verifier, establish significance, or satisfy cost/reporting requirements.
Five CPU tests cover family-specific failures, net versus gross corrections,
insufficient headroom, missing/incomplete records, and own-parent retention.
No pending model outputs were used to write these rules.
