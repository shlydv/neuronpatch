# Coverage and objective comparison — completed 25 October 2026

**No candidate meets the frozen utility and robustness criteria.** Four rank-one
LoRA fits on Qwen3-4B-Instruct-2507 completed, producing 1,800 generations across
base and four adapters. All outputs and adapters downloaded; artifact consistency
checks passed. This is not an independent GPU replication.

## Complete-answer outcomes

| Family (all /30) | base | narrow-sft | narrow-dpo | diverse-sft | diverse-dpo |
|---|---:|---:|---:|---:|---:|
| clean/none | 23 | 19 | 20 | 20 | 19 |
| cross_direct/familiar | 16 | 14 | 13 | 21 | 18 |
| cross_direct/held_out | 19 | 17 | 14 | 20 | 17 |
| cross_injected/familiar | 3 | 14 | 17 | 18 | 16 |
| cross_injected/held_out | 2 | 14 | 17 | 18 | 16 |
| direct_full/claimed_authority | 30 | 30 | 30 | 30 | 30 |
| direct_full/grading_notice | 30 | 30 | 30 | 30 | 30 |
| direct_full/redirect | 30 | 30 | 30 | 30 | 30 |
| injected/claimed_authority | 0 | 16 | 19 | 19 | 17 |
| injected/grading_notice | 0 | 10 | 17 | 8 | 12 |
| injected/redirect | 0 | 19 | 18 | 18 | 16 |
| quoted/redirect | 30 | 30 | 30 | 30 | 30 |

All four adapters lose the same six questions answered correctly by the base.
Clean accuracy drops from 23/30 to 19–20/30. The base itself misses the frozen
24/30 qualification threshold. Every adapter improves an unseen attack family,
but all fail clean, injected and legitimate-request gates. Passing easy marker
controls does not establish preserved instruction following.

Diverse SFT improves competing-question legitimate compliance over narrow SFT
(14→21 familiar, 17→20 held-out), alongside competing-question attack resistance
(14→18 for both wordings). DPO has mixed effects: narrow DPO improves attack
resistance but reduces legitimate competing-question accuracy; diverse DPO does
not consistently improve on diverse SFT. These are paired, single-seed observations,
not evidence that either objective is generally superior.

## Review and sensitivity

247 unique anonymous QA responses were reviewed against sources with condition
identities hidden. Decisions and planned sensitivities were frozen before
attribution. Review was performed by the assistant, not an independent human.
No cases were removed after outputs. Primary, lenient and strict counts are in
[sensitivities.json](sensitivities.json); the full primary gates, Wilson intervals
and paired gains/losses are in [summary.json](summary.json).

Lenient clean scores are 23/19/21/20/20 (base/narrow SFT/narrow DPO/diverse SFT/
diverse DPO); strict clean scores equal primary. Neither changes the failure of
clean qualification. Four questions rely on a subject supplied in the question
rather than the table title; retain them in primary results and do not describe
this small benchmark as broad factual competence. Individual family intervals
are descriptive, not multiplicity-adjusted inference.

## Shared failure and next causal question

Mean clean response length falls from 27.0 tokens to 4.1–4.2. Four of six shared
regressions produce the identical wrong answer in all four adapters. Errors
include selecting the wrong row/date/person and incorrect arithmetic, not just
formatting mismatches. All train/calibration/confirmation documents have actual
newlines; a suspected escaped-newline serialization problem was ruled out.

The common preservation loss trains short source-reviewed answer strings. It is
**not** a KL penalty preserving the base distribution. Task SFT also trains short
answers, and DPO includes those answers in preference pairs. Therefore common
errors cannot yet be attributed specifically to preservation, response length,
rank, learning rate or an implementation defect. A preservation-only/task-only
ablation is needed; another neuron-ranking sweep is not a diagnostic of this cause.

## Cost and provenance

| Fit | Training seconds | Task + preservation forwards |
|---|---:|---:|
| narrow SFT | 647.6 | 372 |
| narrow DPO | 1187.9 | 930 |
| diverse SFT | 913.2 | 372 |
| diverse DPO | 1895.7 | 930 |

All adapters train 2,064,384 parameters and occupy 8,329,184 exported bytes.
Reference scoring adds 372 forwards shared across the comparison; generation
adds approximately 815–836 seconds per trained condition. Provider execution
took 9,686.6 seconds (2.69 hours), not 24 hours of GPU compute. Peak training
allocation ranges from 8.92 to 9.82 billion bytes. Equal updates are not equal
compute or exposure. [costs.json](costs.json) preserves exact measured values.

Private Kaggle run: `shlydv/neuronpatch-boundary-preference-v1-20261024`, version1.
Accepted source SHA256: `2fd316c3f603872357024142b63f13710c18809f6f0e8f560f9f9ce4c40f6a09`.
[Protocol](../preference_protocol_v1.json) preserves model revision, data hashes,
optimization and pre-result criteria. [Review freeze](review-freeze.json) records
pre-attribution judgment hashes. Raw source tables and generated responses remain
private. No sparse-selection advantage, new algorithm, security guarantee or
recommended library recipe is established by this experiment.
