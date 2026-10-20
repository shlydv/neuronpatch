# The update mainly changes the refusal trade-off

The no-training diagnostic completed on the same pinned Qwen3-4B/T4 runtime in
520 seconds. All outputs were downloaded and hashed before releasing the
temporary GPU runtime. Both saved high-dose row patches cold-load through the
public NeuronPatch loader and exactly reproduce all 320 original validation and
utility token trajectories. The restored model fingerprint matches the base.
See [integrity.json](integrity.json) and [the frozen protocol](frozen_protocol.json).

## Observed results

The unprotected patch was scaled to 0.47603 using only calibration prompts. Its
mean UNKNOWN log-odds shift matches the protected patch to 0.60%. The bias control
adds that protected calibration shift (15.44605) to the UNKNOWN logit on the first
generated token of the unchanged base model. There is no further fitting.

| Condition | Supported exact /64 | Missing correct /64 | Old supported exact retained /38 | Utility exact /32 |
|---|---:|---:|---:|---:|
| Base | 38 | 42 | 38 | 31 |
| Protected, full | 36 | 53 | 29 | 31 |
| Unprotected, full | 23 | 57 | 19 | 32 |
| Unprotected, matched | 37 | 50 | 32 | 32 |
| Base + UNKNOWN bias | 35 | 48 | 35 | 31 |

Every setting fails the original joint correction/95%-retention rule. At matched
calibration strength, the unprotected update retains more old correct answers
than protection, but corrects fewer missing answers. Neither dominates the other.
The mean matches one calibration signal, not the validation distribution or the
complete behavior, so this does not prove protection has no useful contribution.

The simple bias gets six more missing answers right while losing three old
exact-correct supported answers. The weaker protected fit in the original pilot
also loses three old exact-correct answers, but gains only two missing answers.
The learned fit additionally improves some answer wording, so the bias does not
dominate every metric. It does establish a necessary, inexpensive control that
was missing from our first comparison.

## What changes internally and in the answers

Against the unchanged missing/answerable labels, first-token UNKNOWN log-odds
AUROC is 0.9080 for the base, 0.8938 for protection, 0.8936 for full unprotected
and 0.8970 for the matched unprotected update. The constant bias preserves the
base ordering exactly. This is a secondary, exploratory ranking measurement
on imperfect labels, without a significance claim. The learned updates do not
improve this measured discrimination on this sample.

On validation, protection increases UNKNOWN log odds by an average 26.71 on
supported questions and 8.44 on missing questions. The matched unprotected values
are 21.57 and 7.82; the bias is 15.45 on both. These shifts are not identical
mechanisms. The learned updates also rewrite answers and occasionally echo the
question. Improving refusal counts alone does not show improved evidence use.

All 13 changed bias continuations are exactly predicted by their first-token
argmax crossing. When the first token stays the same, the complete token
trajectory stays the same. Six changes are missing questions; seven are supported
questions. Of those seven supported changes, three destroy baseline exact matches
and three destroy other correct-content answers in the assistant-only audit.

On that same reviewed subset, protection fixes eight of 12 clear missing-answer
errors but introduces six refusals among 19 correct-content, non-exact baseline
answers. The bias fixes four of those 12 errors and introduces three such refusals.
These are diagnostic subsets, not revised gold or independent human judgments.
All text changes are exposed in [paired_changes.json](analysis/paired_changes.json).

## Decision

Do not promote the current protected-row recipe or keep tuning its rates on this
sample. We have reproduced a working FiNE recipe separately and verified our port;
the present obstacle is achieving conditional behavior without collateral damage,
not an inability to change the model or an unresolved save/load bug.

The next learning objective must reward stronger separation between supported and
unsupported evidence, rather than merely raising refusal preference. Use carefully
checked contrasting contexts for the same question, retain a calibrated no-training
threshold baseline, and measure the whole correction/preservation curve plus actual
outputs. A global UNKNOWN bias must not be able to satisfy the new separation
objective by construction. This is a new hypothesis to test, not a novel method
claim or a working recipe. Neuron selection only earns its place if it outperforms
matched random rows and appropriate low-rank controls under that evaluation.

No additional GPU training was launched after this diagnostic. The broader useful
library goal remains incomplete. This exposed-data result is not fresh confirmation.
