# Eight-task-example comparison: verified results

23 October 2026. **Fisher selection does not show a task-label-efficiency
advantage.** With eight task labels plus twenty preservation labels, selected
LoRA gives 126/128 correct task answers, versus Fisher 121/128 and random
112/128 and 122/128. This completes the bounded pilot, not the overall goal.

## Frozen design

Same Qwen3-4B-Instruct-2507 checkpoint and worker as the 64-label pilot, 64
successful optimizer updates, approximately matched trainable scalar budgets:
LoRA 1,327,104; paired neurons 1,328,640. Fisher discovery uses all eight task
labels and no extra labeled examples. There are only two task training worlds,
one per task family; each supplies all four evidence conditions. This tests
label efficiency within synthetic templates, not broad behavior generalization.

Two learning rates for each method/seed. The [choices](choices.json) were saved
and committed before test inspection, using original-wording validation completion
loss. Confirmation has four fresh task worlds, two known wording variants and a
fifth fresh scope world. The prior 64-label test has different names: comparing
raw counts across those runs is not a paired data-efficiency estimate.

## Full-response results: all rates

| Condition | Selected | Task /128 | Original /64 | Transfer /64 | Scope /16 | Utility /8 |
|---|---|---:|---:|---:|---:|---:|
| base | — | 81 | 37 | 44 | 16 | 8 |
| lora-43-0.0003 | — | 122 | 64 | 58 | 16 | 8 |
| lora-43-0.001 | yes | 126 | 64 | 62 | 16 | 8 |
| paired_random-43-0.0003 | — | 116 | 64 | 52 | 16 | 8 |
| paired_random-43-0.001 | yes | 112 | 64 | 48 | 16 | 7 |
| paired_random-71-0.0003 | yes | 122 | 64 | 58 | 16 | 8 |
| paired_random-71-0.001 | — | 127 | 64 | 63 | 16 | 8 |
| paired_fisher-43-0.0003 | yes | 121 | 64 | 57 | 16 | 8 |
| paired_fisher-43-0.001 | — | 56 | 33 | 23 | 9 | 3 |

Base task accuracy is 81/128 after semantic review, not exact-match 80/128.
One explanatory answer is valid. Three other base answers hit the generation
cap and remain failures; all fitted task responses terminate naturally. Base
scope and utility answers are all semantically correct, including full sentences
that exact matching rejects. The review is agent-performed, with no independent
human adjudicator. All nonexact full responses were inspected; complete exact
reference matches were checked against the frozen labels.

[Explicit decisions](explicit_decisions.json), [all predictions](predictions.json)
and [paired semantic counts](semantic_review.json) preserve the review. The
adjudicator checks input and response hashes and refuses unresolved decisions.

## What the failures tell us

Every validation-selected method gets all 64 original-wording test questions
right. The remaining errors occur under transfer wording, without changing the
format during this experiment. They differ meaningfully:

- LoRA makes two unsupported guesses on missing release evidence. It corrects
  45 base errors, introduces no task regression, and retains all measured controls.
- Random seed 43 guesses on all sixteen transfer missing-evidence questions.
  It corrects 32 base errors but loses one previously correct task answer. It
  also answers “1” to the planet-closest-to-the-Sun question, previously Mercury.
- Random seed 71 makes six unnecessary refusals on supported interview questions.
  It corrects 42 base errors and loses one previously correct task answer.
- Fisher makes five unsupported answers on missing release evidence and two
  malformed supported answers: “I” where Idris is required. It corrects 41 base
  errors and introduces one regression. The [token audit](name_fragment_audit.json)
  shows expected Idris begins with token “Id”; actual “I” is a different token
  followed by EOS. This is not simply a correct token prefix cut short. The
  internal cause is not established by this output-level observation.

The unselected high-rate Fisher fit is substantially worse (56/128, seven scope
regressions and five utility regressions). It over-refuses, confuses names and
sometimes produces malformed outputs. It has no nonfinite-gradient retry, so
this is not evidence of the earlier loss-scaling failure recurring. Finite
training alone does not establish a useful model.

## Selection and stopping decision

The frozen selection-advantage condition fails: Fisher is not eight answers
better than every comparator and is worse than selected LoRA and random seed 71.
The frozen ceiling condition also does not trigger: both random selections are
below 124/128. Neither condition is loosened after observing results.

Validation is a material limitation. All selected methods learn original wording,
but tiny original-wording validation losses do not order transfer behavior.
Random seed 71's unselected rate scores 127/128, versus its selected rate's 122;
random seed 43's selected rate also transfers worse than its other rate. These
are diagnostics, not permission to replace the frozen selections with test winners.
Future validation must represent the intended transfer requirement; these test
inputs are now exposed and cannot be reused as fresh confirmation.

Do not run another formatting or selector variation on this test bank. The
current Fisher recipe has no demonstrated advantage at either observed label
budget. A next method experiment needs a concrete prediction of actual update
benefit and preservation, a validation distribution matching deployment, and a
strong LoRA comparison on a broader task. The current outcome does not justify
promoting Fisher, weakening the goal, or claiming that all neuron selection fails.

## Integrity and cost

Kaggle completed successfully in 2,660.59 provider seconds (44.3 minutes).
All eight fits have zero gradient retries. The [independent verifier](integrity.json)
checks 3,024 decoded outputs, eight full-vocabulary teacher records, adapter
supports and budgets, tokenization, prescribed sample order and training controls.
All downloads completed without failure. No additional GPU run was used for review.

Discovery took 34.42 seconds. Recorded LoRA condition cycles took about 158–159
seconds each; paired-neuron cycles took about 317–333 seconds each. These cycles
include attachment, training, export/reload, merged evaluation, generation and
base restoration, not training alone. They do not show a wall-clock advantage
for this sparse implementation. Selection data, preservation data and evaluation
cost must not disappear from efficiency reporting.
