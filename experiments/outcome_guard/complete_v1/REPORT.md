# Correct greedy answers can hide the same interference that breaks selected patches

**Complete and independently verified. Frozen qualification failed.**
The finite-response guard catches both known selected-patch errors, but rejects
two of four controls previously considered successful from greedy answers.
The new measurements show genuine competition with a wrong continuation behind
those correct answers. They do not justify calling every guard flag a false
alarm, changing the threshold, or claiming an improved trained method.

## Frozen replay

The [protocol](../protocol_v1.json) restores six already trained low-rate
Qwen3-4B endpoints: selected paired neurons, random seed11 and official rank-one
MLP LoRA for each of two exposed counterfactual edits. No optimizer, fresh fact,
new fitting or parameter change is introduced. The model and every original
canonical teacher/generation control match the frozen parent artifacts.

Each case includes its six auxiliary questions plus oxygen; case1 also includes
BBC and Lakers. Accepted finite strings include original aliases, valid formal
UK variants and final-period variants. Two base/reference disagreements are
reported and excluded from retention gates, leaving14 eligible questions across
the two cases. This does not adjudicate those labels or silently correct them.

Qualification prospectively requires each of four LoRA/random controls to retain
accepted-answer mass within an absolute0.02 of the base on every eligible
question, preserve accepted greedy answers and reproduce4/4 requested transfers.
Both known negative rows must exceed the probability-drop threshold and produce
rejected answers. See [analysis](analysis.json).

| Endpoint | Transfer /4 | Greedy retention passes | Maximum eligible accepted-mass drop, percentage points | Probability guard passes |
|---|---:|---|---:|---|
| Case0 selected |4|Yes|0.00073|Yes|
| Case0 random |4|Yes|0.06188|Yes|
| Case0 LoRA |4|Yes|0.06733|Yes|
| Case1 selected |4|No|99.54600|No|
| Case1 random |4|Yes|8.98036|No|
| Case1 LoRA |4|Yes|19.35920|No|

All96 generated observations replay exactly. Full assistant review covers37
distinct prompt/text/truncation contexts. No output truncates or contains
unexpected special tokens. The six endpoints reproduce24/24 requested transfer
answers; their two base controls supply the other eight target observations.
The selected case1 Lakers/O₂ errors remain. See [review](review.json).

## The controls are successful under one requirement and damaged under another

Accepted probability on oxygen's supplied symbol answers changes from0.999954
in the base to0.910150 for random,0.806362 for LoRA and0.004494 for selected
case1. Random and LoRA still greedily emit O. Conditional on the already emitted
O, their competing next-token probabilities are:

| Case1 state | Stop at O | Continue with the subscript ₂ |
|---|---:|---:|
| Base |99.99539%|0.00307%|
| Random |91.01502%|8.87199%|
| LoRA |80.63616%|18.85583%|
| Selected |0.44939%|95.53709%|

These are exact next-token probabilities for the saved direct-prefix execution,
not sampled error frequencies or probabilities of complete O₂ answers. The
unaccepted O₂ suffix states were not evaluated, so later corrections or EOS
probabilities on that branch are unknown. Actual sampling settings also matter.
Nevertheless the same wrong chemical-symbol continuation competes much more
strongly in all three patched case1 models. Greedy correctness alone hid this.

LoRA's accepted Lakers-answer mass also falls from0.999990 to0.935765. Its first
token W rises to6.37065%, versus0.00376% in random and94.81027% in selected. This
does not by itself establish the probability of the full injected target string;
that branch's subsequent logits were not collected.

This differs from the earlier full-vocabulary KL alarm on US alias changes:
the finite-set event largely ignores moving probability from United States to
United States of America. The case1 oxygen flag includes real competition with
₂. Calling the older controls universally clean was too strong; they were clean
on the observed greedy-answer checks, not on this stricter probability contract.

The [post-hoc trie audit](decision_audit.json) separately checks direct-prefix
greedy decisions and their allowed-versus-outside logit gaps. All64 observed
traces agree with cached generation on the tokens actually traced. The greedy
criterion admits all four controls and rejects the known selected mistakes,
but would miss the latent competition above. Conversely, an independent
probability-tree counterexample shows that unchanged accepted-set mass alone
need not preserve the greedy choice. Both axes remain necessary; neither is a
general semantic or unseen-task certificate.

## There is avoidable EOS pressure in our auxiliary training labels

The [objective audit](objective_audit.json) compares two losses on the same saved
base logits: mean token CE for one canonical response, and accepted-response
marginal NLL divided by the **same canonical length**, including EOS. A singleton
accepted set exactly recovers mean CE and its gradient. This avoids a change in
length normalization masquerading as an objective improvement.

For a US auxiliary response, the base already stops after United States and is
accepted by the answer evaluator. The canonical label continues with ` of`.
The resulting gradient with respect to the EOS logit is approximately+0.2
for each of the three US rows inspected. Gradient descent therefore lowers
the EOS logit at an already valid stopping point. With the accepted-set loss,
the corresponding gradients are about−1.29e−7,−1.06e−7 and−8.38e−8.

For the Facebook auxiliary row, canonical mean CE is4.1045 while the normalized
accepted-event NLL is0.00000668. The former regards the base's valid shorter
answer as a substantial training error. This is an objective mismatch we can
identify exactly at the logits. It does **not** prove that those auxiliary
gradients caused the unrelated oxygen failure or identify the responsible
neurons; that requires a matched training ablation.

## Decision and next causal test

Keep the failed0.02 qualification. Do not widen its threshold to admit the old
controls. Also do not discard confidence changes merely because a greedy answer
still passes. The next comparison must report both complete-answer retention
and accepted probability, keeping common requirements for every method.

Before another selector change, isolate the auxiliary-loss mismatch: keep the
original supports, rates,50-update schedule, example ordering, examples and
canonical-length normalization. Compare canonical supervision with accepted-set
supervision only on initially accepted auxiliary rows. Keep the two auxiliary
base/reference disagreements explicitly identified and treated identically
across arms. Leave oxygen and Lakers outside this objective ablation's training
bank so a reduction there would be a genuine transfer observation within this
exposed development study. Existing verified canonical endpoints are the
matched reference; any implementation/control mismatch prevents that comparison.

The hypothesis is that removing unnecessary EOS suppression reduces some of the
unrelated continuation pressure while retaining the requested edits. Failure
would direct attention back to correction gradients or update coordination.
An improvement shared by LoRA and selected neurons would establish objective
repair, not a neuron-selection advantage. The later
[joint-update candidate](../../../docs/joint-update-candidate.md) still needs
strong baselines, fresh cases, multiple budgets and a second task family.

## Integrity and cost

The [private Kaggle version1](https://www.kaggle.com/code/shlydv/neuronpatch-outcome-guard-v1-20261023)
returns0 in840.21seconds,14.00minutes including setup/loading. The worker uses
CUDA0 of the free T4x2 allocation. The requested provider runtime bound is3600
seconds; it was not exercised, so this run does not prove enforcement.
The existing job was never restarted when log-stream connections ended early.

All137695 accepted script bytes and34 bundled files match frozen source revision
`36262363f9d464c78ed322d62f34e4f440b500eb`. Script SHA256 is
`9959047c7907da6730fddfabb5ed0f008b4641890db6ad7de5dad38588652a01`.
The executed submission and protocol also match. Source readback through an
explicit /1 reference returned403; the canonical lookup was independently
checked as version1 before byte comparison. See [source integrity](source_integrity.json),
[numeric integrity](integrity.json), [provider](provider.json),
[completion](completion.json) and [launch](../launch_v1.json).

The verifier independently recomputes probabilities from all840 full saved
vocabulary vectors, all96 decoded answers and all six canonical endpoint-teacher
replays. The12 base-restoration generation assertions and final fingerprint
are execution controls. Trie/objective audits and their tests are postprocessing,
not part of the accepted34-file intervention. No new method or trained benefit
is established by this diagnostic.
