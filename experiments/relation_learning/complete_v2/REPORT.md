# Relation-learning pilot: verified results

23 October 2026. **The behavior is learnable; empirical-Fisher neuron selection
has no demonstrated advantage over LoRA or random selection.** This completes
this pilot, not the library's broader goal.

## Frozen comparison

Qwen3-4B-Instruct-2507; 64 synthetic task examples and 20 preservation examples;
64 optimizer steps for every condition. Rank-1 LoRA trains 1,327,104 scalars;
173 paired neurons train 1,328,640. Fisher discovery uses eight fixed training
examples. Two learning rates per condition, two random supports, one Fisher
support and one LoRA seed: eight fits total. This is not a reproduction of all
FiNE, NeST or SafeNexus settings.

Rates were frozen from validation completion loss before test inspection in
[choices.json](choices.json). Training, validation and test names are disjoint;
half the 64 task-test questions also use new wording. Test questions share
synthetic templates and are not 64 independent demonstrations of generality.

## Complete-response results

| Condition | Validation-selected | Task /64 | Corrections | Regressions | Scope /16 | Utility /8 |
|---|---|---:|---:|---:|---:|---:|
| base | — | 39 | 0 | 0 | 16 | 8 |
| lora-43-0.0003 | yes | 64 | 25 | 0 | 16 | 8 |
| lora-43-0.001 | — | 58 | 19 | 0 | 16 | 8 |
| paired_random-43-0.0003 | — | 62 | 23 | 0 | 16 | 8 |
| paired_random-43-0.001 | yes | 64 | 25 | 0 | 16 | 8 |
| paired_random-71-0.0003 | — | 60 | 21 | 0 | 16 | 8 |
| paired_random-71-0.001 | yes | 64 | 25 | 0 | 16 | 7 |
| paired_fisher-43-0.0003 | yes | 63 | 25 | 1 | 16 | 8 |
| paired_fisher-43-0.001 | — | 61 | 24 | 2 | 15 | 0 |

Corrections and regressions are paired against the same base question. Scope
and utility controls start at 16/16 and 8/8 semantically correct. Exact matching
alone incorrectly marks some of the base model's explanatory answers as wrong.
No held-out answer was truncated. All nonexact full responses were inspected;
complete exact reference matches were checked against frozen labels. There was
no independent human adjudication. The [review ledger](review_ledger.json),
[decoded predictions](predictions.json) and [summary](semantic_review.json)
retain the judgments and response hashes.

## What changed, and what failed

The base makes 16 unsupported guesses and nine unnecessary refusals on the task
bank. Selected LoRA and both selected random supports correct all 25. Selected
Fisher also corrects these 25, but introduces one new refusal: a transfer-wording
release question with sufficient evidence for Uma instead receives UNKNOWN.
This is over-refusal, not failure to suppress an unsupported name.

Selected random seed 71 introduces an unrelated arithmetic error: converting
binary 10 to decimal changes from a correct explanatory answer of 2 to 10.
Selected LoRA, random seed 43 and Fisher retain all eight simple utility answers.
Eight questions cannot establish broad preservation.

The unselected higher-rate Fisher fit emits empty EOS-terminated responses to
all eight utility-test questions (and all eight training/validation utility
questions), while still answering 61/64 task questions correctly. Its gradients
were finite with no numerical retries. This is a real behavioral failure, not
truncation; the experiment does not establish its internal causal mechanism.
Validation loss rejected that rate. Report it rather than hiding instability.

The frozen learning criterion passes: selected LoRA improves by 25/64 task
answers, exceeding the required eight, and loses zero solved-family controls.
The selection-advantage hypothesis is unsupported. A one-question gap does not
establish a general ranking between Fisher and random; there is certainly no
basis here for promoting Fisher as the default.

## Integrity and cost

The private Kaggle v2 job completed all eight fits in 2,908 provider seconds
(48.5 minutes). The earlier failed v1 used 576 seconds. These are job totals,
not per-method speed comparisons. Discovery cost is additional to training.
All fits use the same base, examples, update count and sampling order.

The [independent verifier](integrity.json) checks all 2,880 saved responses,
all eight full-vocabulary teacher records, tokenization, adapter/support
integrity and recorded training controls. The [v1/v2 replay](replay.json)
finds exact equality for the earlier successful LoRA fit's adapter and teacher
tensors, generated tokens, metrics and common training history.

V2 adds bounded loss-scale backoff. The higher-rate LoRA fit has one retry at
step 44, reducing scale 128 to 64 with the same examples before updating Adam.
The other fits have no retries. This resolves that numerical failure without
changing the frozen dataset, rates or number of successful updates.

## Decision

Keep LoRA as the practical baseline. Do not spend another run tweaking formats
or presenting a new selector as progress. This task now has a ceiling at this
budget: it establishes learning, but cannot establish superior efficiency.
A future comparison must ask a useful constrained question, such as whether
selection reaches the same quality at a smaller parameter or data budget,
counting discovery cost and measuring preservation. Freeze that question and
use fresh confirmation data; these reviewed test outputs are now development
evidence. A broader realistic task and replication remain necessary before
claiming a robust library recipe.
