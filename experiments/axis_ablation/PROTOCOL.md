# Intervention-axis ablation — exploratory, 2026-10-19

Use previously exposed preservation_qa data and fixed activation-selected indices.
This is mechanism development, NOT independent confirmation or a paper reproduction.

Question: do gate/up row updates offer a better correction/preservation trade-off
than down-projection column updates, and does explicit preservation alter that?

Factorial: selected columns/rows, random columns/rows (seed 11) and LoRA, each on
one of down, gate, up projections; KL coefficient 0 or 1. Eighteen fits total.
Same model revision, layers 16/20/23, 16 neuron indices per layer, lr .0003,
32 steps, batch size 4, training order and response-token CE. Selected supports
are inherited from the earlier down-input activation ranking, not relocalized for
rows. Random supports use identical neuron indices across axes. This isolates
where weights change for the same support; it does not establish the best selector
for each axis. LoRA rank 2, alpha 2 and same target layers, separately per axis.

Each direct patch has 43,008 parameters. Each LoRA has 34,560 parameters. Match
counts within method across axes; do not claim exact cross-method budget equality.
KL(reference || edited) is applied only on supported-example response prediction
positions, equally weighting examples, as in objective_ablation.

Report every setting and its correction/preservation outcomes; inspect raw errors.
Reload direct patches before evaluation and verify fixed-input logits exactly.
Arithmetic is an already-exposed sentinel. No settings will be declared validated
defaults without independent confirmation. A fixed-learning-rate ablation is not
an optimized comparison of the three axes. No hidden extensions of the search.
