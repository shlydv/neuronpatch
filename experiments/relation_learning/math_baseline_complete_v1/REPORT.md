# Math baseline: do not advance this slice to training

The unmodified Qwen3-4B-Instruct-2507 baseline produced **124 accepted final
answers, one completed mathematical error and three incomplete responses** on
the frozen 128-question GSM8K development slice. This fails the preset requirement
of at least sixteen completed mathematical errors. No adaptation run follows.

## Strict scoring and full-response review

| Category | Count /128 |
|---|---:|
| Completed, strict numeric match to source reference | 91 |
| Completed, correct under reference but rejected by formatting parser | 31 |
| Completed, model calculation correct and source reference wrong | 2 |
| Completed mathematical error | 1 |
| Hit the 512-new-token cap without termination | 3 |

The strict worker initially reported 91 correct, one wrong, 33 needing review,
and three incomplete. The 33 review cases mostly repeat `####` answers or use
that marker in headings. The frozen protocol explicitly requires separating
format failures from mathematical failures; the raw parser result is retained.
This is not a post-hoc relaxation of the mathematical qualification threshold.

All 37 non-exact/full-response cases were inspected by the assistant, not an
independent human judge. The other 91 were accepted by numeric agreement, without
an audit of every reasoning step. The resulting 124 is an accepted-answer count,
not a claim that all reasoning is flawless or that the dataset is error-free.
Every explicit judgment is bound to its response SHA256 in `review.json`.

Two source-label defects matter:

- **Honey, index14:** 800 bees at0.7 liters each produce560 liters. Add the first
  hive's500 for1060, the model answer. The source answer2460 reverses the
  per-bee ratio and inconsistently substitutes700 bees.
- **Marathon, index40:** two-thirds of the speed means three-halves of the time.
  The runners take9,13.5 and18 hours, totaling40.5, the model answer. The source
  answer23 incorrectly scales time in the same direction as speed.

The completed error, **fish index122**, skips daily doubling on the days when
fish are removed or added. It returns63 rather than the reference207. The
question's day-zero convention admits an ambiguity, but the model also omits
doubling on intervening event days; its63 is not the result under either ordinary
convention. Incomplete index96 visibly misreads a layover, while index110 reaches
the correct shirt count and keeps talking. Neither becomes a completed error
under the frozen gate. Even counting all three truncations as errors would remain
far below sixteen.

## Execution and independent verification

Private Kaggle kernel140420814, version1, completed with returncode0 in
2040.32 provider seconds (34.01 minutes), including setup and loading. The
provider exposed two T4s; the worker explicitly placed the model on device0.
No training or weight updates occurred. Provider/worker/internal caps were
3600/3000/2700 seconds. All artifacts downloaded successfully.

The accepted script matched its frozen hash. The downloaded source manifest
matches the submitted bundle. The independent verifier reconstructs all128
prompts with the pinned tokenizer, decodes saved output tokens, checks EOS against
the pinned generation configuration, and recomputes all strict scores and counts.
It loads no model weights. Its token-range check was optimized to read vocabulary
size once rather than once per token; this does not change scoring.

## Consequence

This task/model combination has too little observed mathematical headroom for
the planned comparison. Do not tune away headings, lower the threshold, train on
the exposed one-error subset, or claim sparse adaptation failed here: it was never
run. The broader library goal remains unachieved. The next task must have a
substantive adaptation requirement; repeatedly selecting elementary tasks that
the base model already solves does not test the value of neuron selection.
