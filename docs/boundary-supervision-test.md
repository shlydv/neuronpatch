# Full-document supervision test — preparation, not a submitted experiment

The [completed authority experiment](../experiments/instruction_boundary/complete_authority_v1/REPORT.md)
found a shared failure: every trained adapter ignored all 48 legitimate redirects
after a complete document, although the base followed them. Short direct and
quoted controls all passed. This motivates a supervision ablation before more
neuron-selection development. It does not prove the proposed repair will work.

## Change being prepared

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
assets. Multiple-parent comparator replay, independent verification, final
protocol and submission remain to be completed.

This is a standard counterexample/data-coverage intervention, not a new selector
or a claimed synthesis breakthrough. Equal example counts and optimizer steps
do not equal token or wall-clock cost; record those costs explicitly.

## Required run design before submission

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
  Kaggle T4 job, paid scaling disabled. No new run is submitted by this document.

The causal question is whether replacing these controls repairs legitimate
instruction compliance without sacrificing attack recovery, on fixed supports.
If all representations improve, training coverage was an actionable common
limitation. If only some improve, support/capacity interacts with the repair.
If none improve, inspect the preservation objective, optimization and actual
sample exposure before trying another selector. None of these outcomes alone
establishes a library advantage over strong baselines or the cited papers.
