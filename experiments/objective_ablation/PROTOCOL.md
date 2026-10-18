# Exploratory objective ablation — 2026-10-19

Question: with support fixed, does explicit response-distribution preservation
reduce the over-abstention observed in our previous runs? This is exploratory
reuse of the preservation_qa data and indices, NOT fresh held-out confirmation.

Freeze model revision, data, selected indices, random seed 11, layers, 32 steps,
batch size 4 and learning rate 0.0003. Compare activation-selected columns, random
columns and same-layer LoRA. Sweep forward KL(reference || edited) coefficients
0, 0.1, 1.0. Keep ordinary response-token CE unchanged. Apply KL only to supported
training examples, at every response-prediction position including EOS. Detach
reference logits; average tokens within each example then examples equally. No
constraint on missing-evidence examples. Keeping a base distribution could preserve
base mistakes too; this must be considered before real-document deployment.

Report all nine settings, supported and missing accuracy, over-abstention, arithmetic,
loss components and time. Evaluate exported/reloaded column patches. Do not select
a winner and describe the exposed test set as fresh. This isolates the effect of
one objective within each parameterization, not superiority over paper methods.
If the trade-off improves, independently confirm with new data and tuned baselines.
No claim that this is the FiNE loss: its token scope and normalization differ.
