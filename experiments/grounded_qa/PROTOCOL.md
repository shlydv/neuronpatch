# Grounded QA pilot — fixed before results

A small synthetic feasibility experiment, not a general benchmark or paper reproduction.
Model: Qwen/Qwen2.5-0.5B-Instruct, immutable HF revision recorded in results.
Device: local Apple MPS, float32. No paid compute.

Generate fictional records with counterbalanced missing fields. Every record has one
answerable and one unanswerable question, so entity and template cannot predict the
label. Train/calibration/validation/test entities are disjoint. Training uses parcels
and workshops; final test uses expeditions and exhibits with new field names. The
same instruction is used for all methods. Score free generation by normalized exact
match; also report abstention and save every raw answer. This measures simple
extraction and abstention, not broad knowledge or reasoning.

64 training questions, 16 separate calibration questions, 24 validation questions,
48 final test questions. A separate 16-question arithmetic check is a narrow
collateral-damage sentinel, not evidence of general capability preservation.

Compare prompted base against 16 selected down-projection columns in each of layers
16, 20, 23; layer/count-matched random columns (seeds 11, 29, 47); and PEFT LoRA on
those same projections (rank 2, alpha 2, dropout 0). Report scalar counts; LoRA has
fewer parameters, so this is a same-layer comparison, not exact budget matching.
Selection ranks absolute differences in mean activation at the last prompt token,
using calibration questions only, without gold response tokens.

32 optimizer steps, batch size 4, AdamW, weight decay 0, gradient norm clip 1.
Two learning rates (0.001 and 0.003) for selected and LoRA. Choose by validation
balanced accuracy, then answerable accuracy, then lower learning rate. Use the
selected method's chosen learning rate for all three random seeds. Same shuffled
training order for all fits. No tuning against final test. Greedy generation, 12
new tokens. All final test methods use frozen trained updates.

Success gate: selected improves final-test unanswerable accuracy by >=10 percentage
points over prompted base, loses <=5 points answerable accuracy, and loses <=1 of
16 arithmetic items. If base unanswerable accuracy >90%, this pilot has insufficient
headroom to meet the gate. Evidence favoring selection additionally requires >=5
points balanced-accuracy advantage over the mean of three random controls. Report
LoRA alongside; no claim of superiority from this small pilot. Failure means no
justified utility claim; do not change the task or scoring to rescue it.

Export selected patch, reload onto exact base, check fixed-input logits against the
in-memory patch. Record code/data hashes, versions, counts, timings and raw outputs.

## Execution

Install `pip install -e '.[dev,experiment]'`, then run
`HF_HOME=/path/to/cache python experiments/grounded_qa/run.py` on an Apple GPU.
This first runner deliberately targets MPS. Results go to `results/`; large patch
and probe artifacts are gitignored. Use a fresh output directory for each run.
The script is a bounded experiment, not a general training CLI.

Implementation note before training: GPU activation statistics must be transferred
to CPU before casting to float64. On the tested torch/MPS version a combined transfer
and dtype conversion yielded nonfinite values even though model forwards were finite.
A separate `.cpu().double()` fixes this, with an MPS regression test. Eager attention
is used for all methods. No experimental outcomes were used to choose this fix.
