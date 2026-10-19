# Repeated preservation helps, but does not resolve broad refusal

Six matched replay fits completed locally, 2026-10-19. Supports and data hashes match
the earlier adaptation experiment. No confirmation inference. See REPLAY_PLAN.md.

| Fit | Anchor EM before → after /20 | Supported validation EM before → after /16 | Correct refusals before → after /16 |
|---|---:|---:|---:|
| selected-lr0.0001-kl1.0 | 16 → 16 | 5 → 4 | 5 → 5 |
| selected-lr0.0003-kl1.0 | 10 → 15 | 2 → 4 | 9 → 9 |
| random-lr0.0001-kl1.0 | 18 → 18 | 3 → 3 | 4 → 4 |
| random-lr0.0003-kl1.0 | 16 → 16 | 6 → 6 | 6 → 6 |
| lora-lr0.0001-kl1.0 | 16 → 16 | 3 → 3 | 5 → 5 |
| lora-lr0.0003-kl1.0 | 11 → 12 | 2 → 3 | 10 → 10 |

## Interpretation

At the higher rate, selected-row replay recovers five training anchors and two
supported validation exact matches, while keeping nine correct impossible refusals.
This is evidence that preservation exposure matters at fixed supports and objective
coefficient. It does not validate the localization score: the low-rate selected fit
recovers no anchors and loses one validation match, while both random fits keep the
same discrete validation metrics. Higher-rate LoRA recovers one anchor and one
validation match while keeping ten correct refusals.

The selected high-rate fit still refuses five protected anchors and seven supported
validation questions. Inspecting its training changes shows genuine refusal recovery,
not merely shortening answers. The objective remains a soft preference; repeated
exposure does not guarantee retention. Replay alone is not a robust solution.

Training time was 1.90–2.29 times the matched earlier fits on this machine (excluding
loading/evaluation). This is a schedule/exposure change with added compute, not an
equal-compute comparison. Timing can also vary with machine load.

## First-token diagnostic

Replayed selected high-rate mean separation is 7.137 on training and 5.068 on
validation, versus 6.919 and 4.846 before replay, and 7.526 and 6.319 for the base.
The change partly reduces the damage but does not restore baseline separation.
All six replay fits still have smaller mean pair separation than the base on both
splits. These margins are limited first-token contrasts, not semantic accuracy.
Raw margins and exact last-logit optimization parity checks are in replay/margins.json.

## Design consequence

Checking known-correct behavior throughout training belongs in the toolset, but is
not a sufficient recipe. We have not shown a safe or superior neuron-targeted
adapter. Our LoRA controls have also been restricted to three up projections; they
are not a strong general LoRA baseline. The next reference expands adaptation
across all MLP layers and measures learning over multiple epochs (CAPACITY_PLAN.md).
Its larger budget must be explicit. Establish learnability, then investigate which
parts of a successful update can be localized or shared. Do not keep selecting
neurons for a behavior the current training setup has not demonstrated it can learn.

The original combined-method goal remains open: source-grounded design, stronger
baselines, compression/localization that earns its cost, fresh confirmation, other
tasks/models and a robust reusable workflow are still required.
