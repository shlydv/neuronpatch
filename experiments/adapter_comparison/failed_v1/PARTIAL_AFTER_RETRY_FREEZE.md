# Case-0 evidence inspected after the corrected retry was frozen

This is **partial historical evidence from failed version 1**. Its two completed
case-0 conditions are examined only after corrected version 2 was committed,
uploaded and accepted. They do not tune the retry, choose a global candidate or
replace the registered eight-fit comparison. The retry still starts all fits
from zero with unchanged training settings.

All 159 saved sequences from the case-0 base and two edited conditions decode
independently. The eight saved reference sequences, two final calibration records,
50-step budgets and FP32 scalar counts were already checked separately. No full
qualitative review is claimed by these counts.

| Case-0 result | Saved worst-prompt support | Mean-gradient support |
|---|---:|---:|
| Original requested answer start | 1/1 | 1/1 |
| Untrained-wording requested starts | 3/3 | 3/3 |
| Calibration target leakage | 0/3 | 0/3 |
| Evaluation-subject target leakage | 2/4 | 1/4 |
| Previously exposed scope leakage | 1/8 | 0/8 |
| Lost baseline-reference evaluation prefixes | 0 | 0 |
| Lost baseline-reference utility prefixes | 0/8 | 0/8 |
| Exactly unchanged utility continuations | 4/32 | 6/32 |
| Final worst target NLL | .056956 | .040406 |
| Final worst calibration-token KL | .051661 | .028180 |

The base matches the source reference at the start of only 1/4 evaluation and
8/32 utility answers, and 0/3 calibration answers. Preserving these distributions
is not the same as preserving verified factual knowledge or whole responses.

Both versions train 20,480 scalars in five selected neuron occurrences and share
four of five columns. Mean selection replaces layer7 column3844 with layer0
column1623. The learned updates on shared columns also change; this is not a
single-neuron replacement intervention holding the other updates fixed. The mean
update's float64 occurrence norm is .95633 times the consensus norm, so a dose
confound also remains. No causal role is assigned to the differing neuron alone.

This case provides no advantage for our worst-prompt selection rule over the
simpler mean-gradient control. It is a useful development signal to retain both
controls, not enough evidence to recommend mean selection. Case1, both LoRA
rates, complete-answer review and the full global calibration decision are still
missing from this historical run. No library default changes from this partial
result. See [machine-readable counts](partial_after_v2_freeze.json).
