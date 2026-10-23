# Accepted-answer supervision: partial improvement, selected interference persists

The frozen causal prediction **fails**. Replacing unnecessarily restrictive
auxiliary labels with existing accepted-answer sets reduces oxygen interference
for LoRA and random patches, but does not fix the selected-neuron oxygen error.
All three methods retain all eight requested transfer answers across the two
counterfactual edits. No selected-neuron advantage is established.

## Verified execution

The [Kaggle job](https://www.kaggle.com/code/shlydv/neuronpatch-alias-supervision-v1-20261023)
completed version 1 with return code 0 in 1,787.91 seconds (29.8 minutes).
It used Qwen3-4B-Instruct-2507 and the pinned source in [the manifest](../frozen_v1.json).
Eight fits, 544 generated observations and 1,084 saved prefix vectors passed
[independent verification](integrity.json). Both canonical controls recover the
original histories, weights, teacher logits and generated answers exactly.
Every fit passed export/reload and base-restoration checks. The collector and
postprocessor completed; no replacement GPU job was submitted.

## The intervention contrast

The [frozen protocol](../protocol_v1.json) changes only supervision for five
initially accepted auxiliary answers per case. It accepts the existing aliases
instead of forcing a single canonical string. Supports, target examples, learning
rate, sample order, normalization and 50 updates remain fixed. Oxygen and Lakers
are outside training. Two unresolved reference disagreements still receive the
same canonical loss in both arms. This isolates the specified objective change,
not every possible source of label error or interference.

Probability of the next token being subscript ₂ **after the oxygen prompt and O**:

| Method | Canonical loss | Accepted-set loss | New greedy answer | Halving + answer + transfer criterion |
|---|---:|---:|---|---|
| Selected neurons | 95.54% | 93.34% | O₂ | Fail |
| Random neurons, seed 11 | 8.87% | 2.37% | O | Pass |
| LoRA | 18.86% | 5.41% | O | Pass |

These are conditional next-token probabilities, not sampled error rates.
The prospective criterion explicitly required the selected method to pass;
two successful controls do not turn the failed prediction into a pass.

There is a real selected-patch improvement elsewhere: Lakers changes from
Wassoulou Empire to United States of America. Covered accepted-answer probability
there rises from about 5.18% to 82.46%. The oxygen error remains, with covered
accepted probability only 3.45%. LoRA improves oxygen but lowers Lakers accepted
probability from about 93.58% to 78.75%, while retaining a correct greedy answer.
Thus this is not a uniform preservation improvement.

| Method | Maximum eligible probability drop, case 0 | Case 1 | Both cases meet frozen 0.02 limit? |
|---|---:|---:|---|
| Selected | 0.000092 | 0.965419 | No |
| Random | 0.002406 | 0.024025 | No |
| LoRA | 0.000109 | 0.212509 | No |

The complete values and unchanged literal scoring are in [analysis.json](analysis.json).
All methods retain 8/8 requested transfer answers, but none meets the complete
practical requirements. This probability bank is narrow and does not cover all
collateral errors in the generated-output banks.

## Full-output review

All 136 distinct full-context outputs, representing 544 observations, were read;
[the review ledger](output_review.json) records each assessment and occurrence.
All outputs terminate without truncation or unexpected special tokens.
Capitalization, full UK names, H₂O and complete statements about polygon sides
must not be mistaken for factual errors merely because exact matching rejects them.

The review also prevents calling random or LoRA generally clean. Other-subject
answers change against their source references: for example, the initially
India answers for Raashi Khanna and Shaleen Bhanot become United States in selected
and LoRA patches, and United Kingdom in the random patch. The exposed ATP Finals
answer also remains inconsistent with its Italy reference. These examples are
reported as reference conflicts rather than silently omitted from the result.
Not every historical, biographical or geopolitical reference has been externally
adjudicated. Broadly phrased Russia association questions and the World Cup
reference are particularly poor candidates for unqualified correctness counts.
Frozen metrics remain unchanged; review is a separate interpretation layer.

## What this establishes and what it does not

The objective change causally alters these deterministic matched training
outcomes. It supports the narrower claim that canonical-only auxiliary training
contributes to some interference. It does **not** identify EOS suppression as the
unique mediator, explain the selected oxygen failure, or show selection superiority.
The [gradient audit](gradient_audit.json) is descriptive: cumulative subtraction,
rounding, changed parameter trajectories and Adam prevent treating its norms as
an isolated EOS causal attribution.

Two exposed facts, one seed per condition and a fixed budget are insufficient
for a useful adaptation recipe. We will not automatically launch another selector
or threshold variation on these facts. The next method decision must use a
meaningful behavior task with measured base failures and common-objective LoRA
and random controls. Prior grounded-QA pilots also need to be respected: easy
abstention and threshold shifts are not evidence discrimination. The full library
goal remains unachieved.
