# Real-passage development and confirmation

This is a method-development experiment, not reproduction of a paper. The current
phase measures the unedited model; it does not select a winning adapter.

## Data fixed before predictions

Use official SQuAD 2.0, retaining its CC BY-SA 4.0 license. `prepare.py` verifies
source hashes. Article partitions are disjoint: 40 training-source articles for
calibration, 40 for validation, all remaining training-source articles for training;
the official development articles supply confirmation. Seed 20261019 defines the
article partition independently of predictions.

Sample 64 training, 16 calibration, 16 validation and 32 confirmation paragraphs,
each contributing one supported and one impossible question. Keep full passages;
exclude examples exceeding 512 prompt-plus-target tokens or 24 target tokens. The
paired eligibility and short-context restriction narrow the scope of any claim.
Store every accepted alias, source question ID, title and full context. Samples
are derivatives of SQuAD, not MIT-licensed library code. Public pretraining exposure
remains possible.

## Baseline and diagnosis

Pinned Qwen2.5-0.5B-Instruct, float32 MPS, greedy decoding, 32 output-token cap.
Evaluate training/calibration/validation only. Record truncation, raw outputs,
normalized exact match and token F1, supported-question refusals and impossible-
question answers separately. EM/F1 take the best match across gold aliases. Inspect
mismatches manually before treating all of them as model errors: SQuAD alias lists
are not exhaustive, and valid paraphrases may have low exact match.

Training examples already answered correctly can anchor preservation. Wrong base
answers must not automatically become preservation targets. Calibration may support
localization; validation may guide development. Confirmation must not be run until
the candidate comparison and selection rule are recorded. If subsequently used to
diagnose failures, that confirmation set becomes development data.

## Intended synthesis, not yet an implemented result

The earlier axis experiment motivates up-row control; it does not establish which
neurons to select. Compare candidate localization against random rows and ordinary
LoRA, accounting separately for parameter count and discovery cost. Combine this
with preservation only on reference-correct training behavior. Probe-based selection,
output-aware attribution and shared updates are hypotheses to test individually,
not features to combine without evidence. Loss weights and learning rates require
matched search budgets for controls.

Useful evidence means correcting unsupported or incorrect answers while retaining
correct supported answers. Report the entire correction/preservation trade-off and
paired changes, not just an aggregate score or arbitrary pass/fail label. A claim
of improvement requires fresh confirmation, repeated seeds, and uncertainty; the
broader library goal also requires another task family and model. This baseline
phase cannot establish any such claim.

## First adaptation comparison (fixed before adaptation outputs)

Eight fits: paired output-aware selected up rows with KL coefficients 0 and 1;
random up rows (seed 11) and up-projection LoRA with KL 1. Every configuration
receives learning rates .0001 and .0003, 64 AdamW steps of batch size 2 (one epoch),
identical order, seed 123, zero weight decay, gradient clipping at 1. Direct patches
train 16 rows in each of layers 16, 20, 23 (43,008 parameters); LoRA rank 2 alpha 2
uses 34,560. This is an exploratory equal-step comparison, not matched parameter
count or optimized method comparison. Discovery costs are separately recorded.

Selection uses the 16 calibration paragraph pairs. At the last prompt token, take
the difference of gated activations (impossible minus supported), multiply by the
alignment of each down-projection column with the output-embedding difference
UNKNOWN minus the supported answer's first token, sum over pairs and rank absolute
magnitude within layer. This adapts FiNE's output-aware intuition to paired evidence
support and applies the selected indices to up rows as motivated by the earlier
axis experiment. It is neither FiNE reproduction nor full downstream attribution:
final normalization and later nonlinear layers are omitted. First-token answer
collisions, cancellation and small-calibration noise remain limitations. Selection
must earn its advantage against random control; no causal claim from the score.

Preservation KL(reference || edited) uses gold response tokens, including EOS, only
for the 20 supported training examples with exact-correct base answers. This is a
conservative anchor policy, not an assertion that all other outputs were wrong.
All training rows still receive supervised loss. Calibration and validation outputs
do not enter the preservation mask. Coefficient zero isolates the objective for
selected rows. No confirmation inference in this phase. Save every validation
output and all fits; examine the trade-offs before making a candidate decision.
