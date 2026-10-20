# Protected-row pilot 1: settings fixed before model predictions

This is an exploratory quality comparison, not confirmation or a paper table
reproduction. The exact settings and data hashes are in pilot1_protocol.json and
pilot1_data_manifest.json. The model is the pinned non-thinking
Qwen/Qwen3-4B-Instruct-2507 checkpoint identified in model_candidate.json.

## Data and baseline

Use128 new-passage training examples,32 calibration examples and128 held-out
development examples, each balanced between supported and missing answers.
Article sets are mutually disjoint within this pilot:49 training,12 calibration,
32 validation articles. No paragraph context was used in prior experiments.
Some article topics have historical 0.5B exposure, so this is development data.
The entire official SQuAD development source remains unused for later confirmation.

A stricter unused-article attempt yielded only73 eligible pairs concentrated in
three training articles. It was rejected before any model predictions; the
current split uses new paragraphs and broader article diversity. The rejected
data/preparation source is retained locally. No model outcomes selected the split.

Keep full inputs under512 prompt-plus-target tokens, target length at most24.
No truncation. Training and evaluation use the same pinned chat template and
causal context. Decode greedily, at most32 new tokens, repetition penalty1.0.
Score supported answers against all SQuAD aliases, missing answers as UNKNOWN.
Inspect raw errors because valid alternate wording can miss exact-match scoring.

Measure32 separate utility prompts: arithmetic, uppercase conversion, antonyms
and capitals. Uppercase scoring is case-sensitive. These prompts do not enter
training, calibration, neuron discovery or hyperparameter selection.

## Protection and selection

Choose up to four baseline-exact supported calibration responses from distinct
articles, sorted by total length then ID. Each is at most400 tokens; the combined
budget is1600 and at least two anchors are required. Protect the complete prompt
and actual generated token trajectory, including its emitted terminator. All
relevant unpadded inputs are captured at up projections23,29,34.

Use the FP64 numerical input span, with machine-precision rank tolerance. No
statistical truncation. Report ranks, remaining directions and residual errors;
an infeasible span stops the run. This is a finite calibration constraint, not
a guarantee on new prompts or exported FP16 weights.

Rank output rows within each fixed layer by the norm of the mean calibration CE
weight gradient after input-span projection; select32 per layer. This measures
update suitability under the proposed constraint, not a causal neuron label.
A seed42 uniform without-replacement random selection matches the row/layer counts.

## Eight fits

Compare protected selected rows, the identical unprotected rows, protected random
rows, and rank1 LoRA on all36 up projections. The LoRA baseline intentionally has
a stronger parameter budget:442,368 parameters versus245,760 allocated row
parameters. Row effective linear dimensions are also reported. Equal parameter
count is not claimed.

Every family receives two rates: rows0.0003/0.001; LoRA0.0001/0.0003. There is one
epoch of160 adaptation steps, combining all128 training and32 calibration rows.
Every control receives those same160 labels, including those used for discovery.
Each step also replays one actual baseline-correct
anchor response, cycling the same anchors, with equal example loss weights.
Use AdamW, zero weight decay, global gradient clipping1, training seed123 and the
same shuffled correction order. Report gradient-scaler skips; no silent refits.
Base models remain in eval mode with eager attention. No checkpointed recomputation
is used because scoped forward hooks would need a different lifetime.

Evaluate exported/merged weights on validation, utility and anchors; also record
live anchor predictions. Saving/export precision can weaken the numerical
constraint. Restore original weights and check their exact fingerprint and one
baseline generation after every fit. The run has a7200-second budget, including
model loading and inference after environment setup.

## Interpretation

Report all settings and correction/preservation trade-offs, including no edit.
Within each family select validation balanced EM among settings retaining at
least95% of baseline-correct supported validation answers; otherwise report its
Pareto settings. Lower learning rate breaks ties. Utility does not select rates.

A promising candidate should gain at least5 percentage points on missing-answer
EM over the base, retain at least95% of baseline-correct supported and utility
answers, and improve the correction/preservation trade-off against the row and
LoRA controls. These are promotion criteria, not a reason to hide failed outputs
or abandon learning from a negative result. A ceiling baseline limits the claim.

Any apparent advantage needs fresh confirmation, multiple seeds and another task
and model. No such conclusion follows from this pilot alone. Source/code/data,
discovery time, training time, parameter counts, memory and export size must stay
visible.
