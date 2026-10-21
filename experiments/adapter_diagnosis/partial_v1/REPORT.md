# One-hour bound reached after sixteen of eighteen interventions

**Historical partial checkpoint.** The [complete frozen grid](../complete_v1/REPORT.md)
is now verified, including the two missing conditions. The timeout and partial
evidence below remain unchanged as the original execution record.

The original worker is terminal **ERROR**, return 1, because our 3600-second
runtime bound raises `TimeoutError` before the next intervention. Kaggle ran
the requested job; this is not a queue or free-plan failure. The wrapper took
3670.59 seconds on allocated Tesla T4s, using CUDA0. Model loading, exact replay,
generation and repeated full-weight restoration checks all used this budget.

Sixteen interventions finish and restore the base. Only case 1's
`only-final-three` and `without-final-three` remain unrun. The original source,
conditions, data and fitted factors remain frozen. No method or learning rate
is changed to repair this execution shortfall.

## Verified partial evidence

The independent CPU verifier checks **954/954 decoded observations**, exactly
replays all 212 base/full control observations against the completed parent,
and reconstructs 18 calibration records including the two baselines. It checks
the completed records form the exact prefix of the frozen execution order,
not an outcome-selected subset. No completed-run marker exists; the verifier
explicitly reports `complete:false` and names the missing suffix.

See [integrity](integrity.json), [provider](provider.json),
[interventions](interventions.json), [controls](controls.json),
[partial scores](summary.json), [decisions](decisions.json) and
[execution log](execution.log). The accepted 220032-byte source remains
SHA256 `7fb054bfb65908f9f820c04d36add34475657bcde8f442deb59b0ed99590c317`.
All 61 bundled file hashes and the downloaded submission identity agree.

## Completed projection-family comparisons

Both edits have all seven projection-family conditions below. Only the
separate final-layer pair lacks its second edit; those rows have explicit
smaller denominators in the partial scores and cannot pass as complete.

| Retained learned updates | Original starts | Untrained wordings | Evaluation replacement leaks | Exact utility continuations |
|---|---:|---:|---:|---:|
| Full adapter |2/2|6/6|0/8|20/64|
| Attention only |1/2|2/6|0/8|42/64|
| Incoming + outgoing MLP, no attention updates |2/2|6/6|0/8|26/64|
| Incoming MLP only |1/2|3/6|0/8|48/64|
| Attention + outgoing MLP |2/2|5/6|0/8|25/64|
| Outgoing MLP only |1/2|1/6|0/8|31/64|
| Attention + incoming MLP |2/2|5/6|0/8|40/64|

Base attention computation remains present in every condition. We remove
**learned updates**, not base components. The observed target starts do not
require the fitted attention updates in these two cases. Keeping the fitted
incoming/outgoing MLP updates together is substantially stronger than keeping
either family alone. This is a reason to test paired parameterizations, not
proof of a clean input gate, a unique circuit or an optimal trainable subset.
Groups are co-trained and have unequal sizes and norms.

The MLP-only subset still fails the registered strict training-confidence
bound: case 1 worst target-token NLL is .14139 (minimum target-token probability
about 86.8%), versus the required 90%. Its six successful untrained starts remain
real observations; this threshold failure does not erase them. Full adapter
alone passes all available-case training/calibration count criteria so far.
Report the separate measurements instead of turning either one into a global
success/failure claim.

All completed conditions retain baseline-reference evaluation/utility starts,
but the parent base is reference-prefix-correct on only 8/32 utility questions.
Exact continuation agreement is not factual quality. All available capped
continuations are now reviewed as 227 distinct contexts: 145 are exactly equal
to previously reviewed prompt/reference/output contexts, and 82 are newly read.
See the [review index](review.json). This is assistant review, not blinded human
factuality labeling; the missing 106 observations remain unavailable.

The MLP-only subset's six successful starts conceal a material failure: it
reintroduces false generated personal allegations on a corresponding training
prompt where the full adapter produces a different, still fabricated biography.
Those allegations are not republished. Other subsets also repeat or invent
details after the target. Therefore removing learned attention updates preserves
the observed target starts, but does not establish preservation of the full
response. This is not a clean correction-versus-preservation component split.

All inspected target-start misses are baseline countries or generic repeated
completions, not clear aliases for the requested counterfactual target.
A post-hoc literal scan finds no new target substring in unrelated banks for
the completed conditions. That still does not establish unrelated-answer
factuality. No condition is promoted as a robust editor, and no fresh
confirmation data is consumed.

## Finish the fixed grid without repeating the completed ablations

The [resume manifest](../resume_v1.json) hash-binds all verified partial artifacts.
A separate finisher must regenerate both edits' 212 base/full control
observations, copy the 14 verified completed nonfull interventions unchanged,
and infer only the two missing conditions (106 new observations). That is 318
new observed continuations and zero new fits. The original protocol is byte-
identical. Combined outputs carry explicit source-run provenance and require
an independent verifier supplied with the frozen resume manifest.

CPU tests compare a complete tiny-model run against an interrupted-prefix
resume: all 18 intervention records agree, only the missing pair plus full
controls executes again, and changed copied bytes are rejected. This checks
the resumption mechanism, not full-size behavioral quality. The full diagnostic
and library/method goal remain incomplete until the required work is verified.

The first finisher submission was cancelled while queued after Kaggle rejected
the failed run as an input source. A ready **private** checkpoint dataset now
provides the same 44 files, with matching names and sizes. The worker requires
their exact recorded hashes before model loading. The corrected version2
source is byte-identical; only its input metadata changes. See the
[delivery record](../finish_delivery_v2.json) and [current launch](../launch_finish_v2.json).
