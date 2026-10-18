# The failure is already visible on training examples

Saved-adapter diagnosis, 2026-10-19. No retraining or confirmation inference.

| Fit | Train supported EM /64 | Train correct refusals /64 | Anchor EM /20 | Anchor refusals /20 |
|---|---:|---:|---:|---:|
| base | 20 | 17 | 20 | 0 |
| selected-lr0.0001-kl1.0 | 18 | 39 | 16 | 3 |
| selected-lr0.0003-kl1.0 | 12 | 58 | 10 | 10 |
| random-lr0.0001-kl1.0 | 18 | 21 | 18 | 2 |
| random-lr0.0003-kl1.0 | 16 | 31 | 16 | 4 |
| lora-lr0.0001-kl1.0 | 16 | 29 | 16 | 4 |
| lora-lr0.0003-kl1.0 | 11 | 57 | 11 | 9 |
| selected-lr0.0001-kl0.0 | 18 | 40 | 16 | 3 |
| selected-lr0.0003-kl0.0 | 7 | 59 | 6 | 14 |

All eight reloaded adapters reproduce every previously saved validation output
exactly (256/256); the base reproduces all 192 development outputs. Serialization
is not an explanation for these observed failures.

## Learning versus transfer

The low-rate selected/KL fit decreases supported training EM from 20/64 to 18/64
while increasing correct impossible refusals from 17/64 to 39/64. Supported training
refusals rise from 12/64 to 29/64. It retains 16/20 exact-correct anchors. Three lost
anchors become UNKNOWN (1977 church merger, 1989 CBS acquisition, 1891 endowment).
The fourth is only wording: "approximately 20 to 25%" versus "20 to 25%". Thus at
least three protected examples genuinely break even on training data.

This rules out a *pure* held-out transfer explanation. It does not establish that
conditional learning is impossible, or that the model memorized no useful examples.
For example, both questions in 10/64 training pairs become exact-correct versus
5/64 for the base. Some useful changes coexist with substantial broad refusal.

## Margin evidence

For each same-passage pair, compare UNKNOWN's first-token logit against the same
supported-answer token. The low-rate selected/KL fit raises this margin by 3.81 on
supported training questions and 3.40 on unsupported questions. On validation it
rises by 4.62 and 3.78 respectively. Mean pair separation therefore *decreases* by
0.42 on training and 0.84 on validation. All eight fits decrease this mean separation
on all three development splits. See `diagnostic/comparison.json`.

This supports a common refusal shift rather than improved average separation under
this diagnostic. It is not a complete causal explanation or a reliable answerability
classifier: the supported answer may be irrelevant to the other question, other
candidate tokens compete, UNKNOWN casing can differ, and only first-token logits
are measured. Better full-response discrimination cannot be ruled out solely by
this statistic, although the actual new refusals show real damage.

## What to fix next

The current preservation term is soft and appears in only 18/64 training steps.
It protects anchors when they occur in a batch, not against every later update.
Next test replayed anchors throughout training while holding supports, supervised
order and optimizer fixed (REPLAY_PLAN.md). This changes preservation exposure and
adds compute; report both rather than crediting neuron discovery for an objective
change. Anchor replay is established practice, not proposed novelty.

If anchors are protected but corrections disappear, investigate conflicting update
directions and conditional update suitability. If even training anchors remain
broken, resolve the objective/optimization before claiming localization is the
main bottleneck. Broader preservation examples and semantic scoring still matter.
The goal remains a useful combined adaptation method, with competitive baselines
and fresh multi-task/model evidence—not a better refusal score on this dataset.
