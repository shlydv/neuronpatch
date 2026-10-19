# Next objective comparison: protect the largest token-level change

Exploratory hypothesis: averaging KL over response tokens underweights a large
answer/refusal decision change when the remaining teacher-forced continuation
remains close to the reference. This follows measured token losses, not a claim
that the loss is novel. FiNE's worst-token NLL is a related optimization design
lesson; worst-token reference KL is our distinct adaptation and must be tested.

Keep the 512 training examples, 115 reviewed anchors, all-MLP rank-2 LoRA, seed,
supervised order, 256 steps, two replay examples per step, and two learning rates.
Use the existing mean-KL coefficient-1 coverage fits as controls. Train two new
objective arms at both .0001 and .0003:

1. Mean response KL with coefficient 5.
2. Maximum response-token KL per example, averaged over examples, coefficient 1.

Five is an explicit strength control motivated by the roughly 4–6 first/mean
ratios in refused training examples. It is not exact gradient- or penalty-matching
and must not be described as such. The same two-rate search budget applies to both
new arms. Keep teacher forcing, target masks and KL direction unchanged. Do not
change neuron selection or add cluster sharing in this comparison.

Use explicit greedy decoding with repetition_penalty=1 for every compared model.
Neutral validation predictions for both existing controls are already saved in
decoding_512. Generate comparable neutral training-anchor responses for the two
controls before comparing training preservation. Retain original decoder results
as historical evidence, not interchangeable control scores.

Primary mechanism check: reduced first-token/max KL and retention of valid answers
on the same 115 anchors. Inspect all changed anchor outputs, not just EM gains.
Primary utility check: development supported-answer quality and appropriate
missing-evidence refusal jointly, including base-correct retention and raw output
inspection. Training-only gains do not establish useful adaptation.

If max KL and stronger mean KL behave similarly, report that stronger preservation
helps; do not credit special token targeting. If both fail, investigate examples
and gradient conflict instead of adding another selector. If max KL offers a
better observed trade-off, it remains a hypothesis requiring seed replication,
fresh confirmation, matched neuron/random/LoRA comparisons and another task/model.
No single pilot completes the combined-method goal.

Implementation: `preservation_kl` accepts `token_reduction="max"`; the default
remains `"mean"`. Numerical tests cover both KL directions, masked NaNs, unequal
valid lengths, detached references, tied maxima and identity gradients. The
runner's `--objective-variant mean5` / `max1` options require the reviewed 512
prefix, set neutral decoding explicitly, and archive the exact loss implementation.
The two invocations were launched serially on 2026-10-19, mean5 first; each contains
both rates. Results go to token_loss_mean5/ and token_loss_max1/. A running first
invocation does not establish that the second has begun or that either has finished.

After the four-fit process finishes, `evaluate_decoding.py --training-controls`
will regenerate the two coefficient-1 coverage controls on the same 512 training
examples under neutral decoding. This is read-only evaluation into a distinct
token_loss_controls/ directory. Until then, do not compare new neutral training
scores with old penalized-decoding training scores.

`analyze_token_objectives.py` prepares the development comparison against the
already neutral controls and a direct max1-versus-mean5 comparison at each rate.
It checks data, anchors, their order, replay exposure, target modules and decoder
metadata before comparing outputs. Base-correct retention is defined using the
neutral base's answers; it must not reuse the old decoder's 45-answer cohort.
