# Earlier stopping preserves the edits, but does not fix guided interference

**Complete and independently verified.**
The seven-fit Kaggle study completes successfully. Selecting the first adequate
checkpoint reduces fitting steps and recovers the deliberately unstable
high-rate edit. It does not remove the selected low-rate patch's two clear
unrelated factual errors. Early LoRA and the random control learn the same
requested transfers without those observed errors. The current guided recipe
does not pass its advancement criteria.

## Frozen comparison

The [protocol](../protocol_v1.json) uses the same two exposed development facts,
three correction responses, six auxiliary/preservation responses, selected
supports, optimizer and approximately1.33million trainable scalars as the
[original comparison](../../paired_mlp/complete_v1/REPORT.md). These are
counterfactual test edits, not statements of real-world facts.

For guided neurons, random seed11 and official MLP LoRA, select the first
checked post-update checkpoint where every token of all nine training responses,
including EOS, has probability at least0.9. No held-out answer selects a
checkpoint. Continue each trajectory to50 solely to reproduce its original
endpoint. The known failing high-rate case is a separate diagnostic, not an
additional candidate for tuning.

## Complete-answer outcomes

Each comparison row aggregates two facts and four transfer wordings per fact.
The49 preservation observations were accepted by the frozen scorer before
editing;54 preservation observations were generated in total.

| Condition | Selected updates, cases0/1 | Early transfer /8 | Early scored preservation losses /49 | Full-step losses /49 | Clear unrelated factual errors at early checkpoint |
|---|---:|---:|---:|---:|---:|
| Guided paired neurons |12 /12|8|3|4|2|
| Random paired neurons, seed11 |30 /24|8|0|0|0|
| Official rank-one MLP LoRA |30 /24|8|0|1|0|

The guided early checkpoint still substitutes the edited target country in the
unrelated Lakers question and changes the chemical symbol for oxygen from O to
O₂. The third scored loss is a valid full UK name absent from the frozen alias
list. Its50-step endpoint has the same two factual errors and two UK alias
misses. LoRA's sole50-step scored loss is also a valid UK alias; earlier stopping
changes its wording without establishing a factual-quality improvement.

Both guided cases fit in fewer updates. That advantage concerns the training
responses, not safe parameter localization. In the interfering second case, the
nominal update norm falls from2.138 at50 steps to1.490 at12, but both factual
errors remain. Early stopping and a smaller update are insufficient here.

All434 generated observations terminate with EOS and have no unexpected special
tokens. Full review covers67 distinct prompt/reference/output contexts:66 exactly
match previously reviewed contexts, and one is new. The new context is the
high-rate early patch's incorrect largest-planet answer, Saturn. Review is by
the assistant, not a blinded human assessment. Frozen labels remain unchanged.

## Cost includes the working implementation

| Condition | Training updates summed | Observed early adaptation prefix, seconds | Fixed50-step reference, seconds |
|---|---:|---:|---:|
| Guided paired neurons |24|144.89|179.61|
| Random paired neurons |54|154.28|170.15|
| Official MLP LoRA |54|37.84|56.41|

The early prefix includes attachment, discovery, training, all preceding complete
response checks/raw checkpoint writes and the selected export. It excludes
common model loading, deployment scoring and research diagnostics. The fixed
reference counts training, one final check and export. These are sums of observed
components within continued control trajectories, not independently timed
production runs or replicated timing estimates.

Guided training itself takes11.81seconds across both cases, versus25.90 for random
and28.96 for LoRA. Guided discovery adds8.96seconds. However, sparse attachment
and export each take about30seconds per case; both contain strict full-base
fingerprinting. LoRA attachment/export are below one second each. The individual
hashing calls were not profiled separately. Thus the current implementation's
total guided prefix is about3.83times LoRA's despite fewer optimizer updates.
Removing integrity costs from the headline would misrepresent this workflow.

## A late update causes premature stopping in the high-rate diagnostic

The deliberately failing first-case high-rate patch passes its first confidence
check at12. That early patch transfers4/4 requested answers, versus1/4 at50, and
has2 scored preservation losses rather than10. It still changes another
person's country answer and answers Saturn for the largest planet, so this is
not a clean candidate.

At the exact final sampled correction prompt, mean NLL rises from0.00002922
immediately before update50 to1.196853 afterward. Grad-enabled and no-grad
logits are exactly equal at the measured pre-update state. The saved evaluations
also agree after the update; this particular failure is an optimizer-step
effect rather than a switch in autograd execution mode. The earlier
[live/merge replay](../../paired_endpoint/complete_v1/REPORT.md) already rules out
merging as its origin.

The target word tokenizes as `Sy` followed by `ria`. The first token retains
probability above0.9999. At the second position, `ria` falls from0.9999975 to
0.0275852, while EOS rises from0.00000133 to0.972413. The model therefore emits
the beginning of the target and then stops; it has not simply switched to a
different full answer.

The [offline Adam decomposition](../step_analysis.py) reconstructs the recorded
update from parameters, clipped joint gradients and previous moments. At the
same recorded second-moment denominator, the correction-gradient dot products
are:

| Contribution to the update | Dot with correction-only gradient |
|---|---:|
| Current joint-gradient term |-8.15986e-6|
| Historical first-moment term |+1.45429e-5|
| Actual complete update |+6.38306e-6|

The historical term outweighs the current term's local descent on the correction.
The actual update still has a negative dot with the clipped joint gradient,
-1.26324e-4. These are local autograd directional measurements. They do not
quantitatively explain the much larger nonlinear NLL jump or prove that resetting
momentum, changing optimizers or deleting a particular neuron would fix it.
The first moment contains past correction and auxiliary gradients, not a unique
source of interference.

## Verification and decision

The [Kaggle version1](https://www.kaggle.com/code/shlydv/neuronpatch-paired-checkpoints-v1-20261022)
returns0 after2808.97seconds, about46.8minutes. Accepted source is159707bytes,
SHA256 `c40f1fe3616287f1b70a3535ce83c66859ec4ca371db911c1bb56fcb368825aa`.
All36 bundled file hashes, the downloaded execution submission and protocol
match the accepted source;44 independently downloaded JSON files are byte-identical.
The [independent verifier](integrity.json) checks434 decoded observations,107
teacher records, all seven original final histories/parameters/logits and the
recorded Adam step. All parent final generation controls are exact. All14
selected/final grad-enabled/no-grad comparisons have identical logits over446
response positions. See [source integrity](source_integrity.json),
[provider](provider.json), [completion](completion.json), [review coverage](review.json),
[analysis](analysis.json) and [step decomposition](step_decomposition.json).
The token-transition/decomposition postprocessing passes six focused tests under
the pinned worker runtime; file hashes are in [postprocessing provenance](postprocessing.json).

Keep this guided recipe experimental. Do not spend fresh confirmation cases
scaling it unchanged: its two genuine preservation errors survive earlier
stopping, and its observed total cost does not beat LoRA. Stopping is useful
optimization control, but it does not establish safe update directions.

The next method hypothesis must measure correction gain together with functional
sensitivity on representative preservation inputs before allocating its update
budget, and enforce measured preservation during fitting. Compare the same
protection/stopping policy for guided supports, random supports and LoRA, with
discovery and integrity costs exposed. A gradient ratio or sparse parameter
format alone is not a new contribution; the value must be demonstrated through
a better correction/preservation/cost tradeoff. No such advantage is established
by these two exposed development facts.
