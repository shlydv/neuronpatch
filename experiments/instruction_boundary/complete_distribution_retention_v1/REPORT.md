# Matched-trajectory distribution retention — 25 October 2026

**KL reduces clean regressions but sacrifices learned behavior. This control is
not a successful repair or an advantage for neuron selection.**

| Condition | Clean /30 | Authority attack /30 | Competing attack /30 | Legitimate competing task /30 | Base-correct clean losses |
|---|---:|---:|---:|---:|---:|
| Base |23|0|2|19|0|
| Previous SFT + hard-label preservation |20|19|18|20|6|
| Same SFT + distribution retention |20|6|13|17|3|

The previous recipe lost six correct clean answers but gained three. KL loses
three and gains none. It repairs four of the previous six losses (arithmetic,
ruler identity, preceding chapter, political party), retains two (last date,
first wrestling opponent), and introduces a gold-medal matching error. Thus the
unchanged 20/30 total hides a different set of successes and failures.
Mean clean answer length returns from 4.13 to 24.20 tokens (base 27.03), consistent
with better retention of response style. This does not establish that length
causes correctness or that KL preserves all relevant computation.

The frozen checkpoint observations are 21/30 with two regressions at step 31 and
22/30 with one regression at step 93. Neither reaches the original 24/30 clean
threshold. No checkpoint is selected; attack and legitimate-task evaluations
were performed only at the final endpoint. We cannot infer a successful combined
tradeoff from the intermediate clean answers.

## What changed and what this establishes

One rank-one LoRA fit on Qwen3-4B-Instruct-2507 changed only the clean preservation
objective: hard-reference NLL became KL(base || edited), weight 1. The same 31
short reference trajectories, 186 task examples, seed, schedule and parameterization
were retained. This controls the objective comparison, but does not test retaining
the base's complete generated trajectories or all prompt contexts.

Initial KL was exactly zero for all 31 cached examples. There were 186 successful
steps and no gradient retries. Mean KL across successive training ranges was
0.221, 0.112 and 0.094; those are changing pre-update training measurements, not a
held-out drift metric or a bound on generated-answer errors. Low training loss
does not certify behavioral preservation. This single weight does not establish
that all KL recipes fail, nor establish the best achievable tradeoff.

## Integrity and review

[Kaggle version 1](https://www.kaggle.com/code/shlydv/neuronpatch-distribution-retention-v1-20261025)
completed with return code zero. All outputs downloaded without failures.
Accepted source SHA256: `d5a5c0130ea993cf82b7880ef076186290e8793a876b71a9f35bb885b5995449`.
The CPU verifier checked protocol/data identities, prior-adapter exact replay,
reference-cache hashes and shapes, training schedule and losses, finite adapter
weights, export receipts and all 300 generations. This is artifact consistency,
not independent GPU replication. EOS termination was not independently recomputed.

The final and intermediate packets contain 194 unique source-bound responses.
158 reuse identical frozen judgments; 36 new responses were source-reviewed with
arm identities hidden (10 correct, 26 incorrect). Decisions were frozen before
attribution. Reviewer: assistant, not independent human. All 30 documents remain.
The Don January and Cello tables omit contextual headers; their previous frozen
interpretation remains in force. No new exclusion or favorable rescoring is used.
These documents are exposed development data; no fresh generalization claim is
supported. The base's 23/30 qualification failure remains a limitation.

## Cost and decision

Provider execution took 2160.36 seconds (36.01 minutes). Fit time was 1066.94
seconds including 171.53 seconds of intermediate generation; final evaluation
was 267.47 seconds. Peak allocated training memory was 9,804,741,120 bytes.
The adapter contains 2,064,384 trainable scalars and exports 8,329,184 bytes.
Reference caching and scoring are additional costs; see raw manifests and costs.

This comparison supplies a useful diagnostic: preservation objectives change
both collateral damage and task acquisition. It does not supply a working
baseline on which to declare guided neuron selection superior. Do not start a
selector sweep or another formatting variant from this result. Before another
GPU run, examine the concrete remaining failures and whether the supervision is
asking the model to rewrite correct answer behavior unnecessarily. A subsequent
candidate must make a falsifiable prediction about the behavior/cost tradeoff,
then be compared with tuned LoRA and matched random supports on fresh inputs.


## Post-review failure separation

A descriptive audit of the frozen outputs separates explicit attack compliance
from answer correctness. These are behavioral signatures, not traced circuits.

| Condition | Exact authority-marker responses /30 | Correct authority answers /30 | Wrong authority answers identical to its own clean answer |
|---|---:|---:|---:|
| Base |30|0|0|
| Previous combined SFT |0|19|6|
| KL retention |21|6|0|

The previous recipe's 11 authority failures are all non-marker outputs, and six
exactly repeat its clean errors. The KL recipe's 24 authority failures include
21 exact attack markers. Therefore the KL change weakened the learned authority
boundary in this test, not merely factual answer accuracy. Conversely, failing
to print the marker does not prove correct boundary reasoning or general safety.

For competing questions, the previous SFT has 12 wrong injected answers, six
identical to its clean errors. KL has 17 wrong injected answers, none identical
to its clean output; one exactly equals the competing answer. Another answer
(Derrick Williams) is consistent with answering the competing weight question
incorrectly, illustrating why exact-alternative counts underestimate wrong-task
behavior. It is not legitimate to label all remaining wrong answers as either
routing failures or factual failures automatically. The saved exact signatures
are descriptive and overlapping; no causal neural mechanism is established.

This changes the next design question: preserve the model's answer computation
while teaching where instructions are authoritative. Merely raising a global
clean-retention weight can preserve the original vulnerability. A candidate worth
examining is clean-to-attacked **distribution matching on a shared continuation**,
with matched legitimate updates using the appropriate donor teacher. That differs
from self-KL on clean contexts: it supplies a positive learning signal on attacked
contexts without forcing every task into short reference wording. It is ordinary
cross-context distillation as a candidate, not a novelty claim or scheduled run.

Earlier experiments already copied hard teacher answers, including mistakes.
Repeating that is insufficient. Before implementing or launching this candidate,
check prior methods/code, verify source-correct complete teacher trajectories and
causal token alignment, and specify how teacher errors and full reference-cache
costs are handled. Compare against the current NLL and KL controls; do not change
neuron support simultaneously. Its key falsifiable prediction is reduced exact
marker compliance without losing the clean QA behavior recovered by KL. If it
only retains clean behavior while continuing to print markers, it fails that
prediction. A useful result still needs fresh wording/documents and stronger
baselines before any claim for neuronpatch.
