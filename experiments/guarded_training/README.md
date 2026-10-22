# Measured preservation during fitting

Experimental implementation of the next method candidate. **No pretrained
result or advantage over LoRA has been established for this loop.**

The [completed direction calibration](../functional_selection/complete_v2/REPORT.md)
found a useful gain/risk ranking but unreliable finite quadratic risk prediction.
This candidate uses risk for initial support ranking and actual measurements
for accepting updates. It changes the method rather than relaxing that study's
failed criterion.

For the same correction and preservation examples, fit every representation with
equal-example full-response task cross entropy plus base-anchored KL. Propose a
standard Adam step, then measure the complete training loss and exact
KL(base||current) on supplied preservation contexts. Require improved correction
loss and both mean and worst-token preservation limits. On rejection, restore
parameters, gradients, Adam moments, step counters, rates and RNG; retry the same
proposal at a smaller learning rate. Exhaustion commits no change.

The KL gradient uses the equivalent log-sum-exp Bregman form, with zero gradient
at the exact reference. Acceptance separately computes probability-distribution
KL in FP64. A confidence stop checks every supervised correction token,
including EOS; it cannot inspect held-out completions.

## Current checks and limits

CPU checks compare an accepted retry with a separately executed Adam update;
test full rollback after exhaustion and exceptions; independently check the
anchored loss; fit a separable response while preserving reference logits; and
return no change when the correction conflicts with a zero preservation budget.
These establish plumbing and mathematical behavior, not LLM adaptation quality.

Only standard Adam/AdamW without closures, hooks or differentiable stepping are
covered by the transaction. The model must be in eval mode and its original
weights and buffers must remain immutable. The transaction is not a rollback
mechanism for arbitrary external state. FP32 adapter parameters are explicit;
the model base may use another supported floating-point dtype.

Finite checks protect the supplied conditional contexts. They do not certify
arbitrary generations, unrepresented tasks, or behavior after export/merging.
Canonical-answer contexts and cached base-generation contexts differ; that
choice must remain explicit. A no-edit result is permitted and is not counted
as successful adaptation.

## Next pretrained comparison

Use identical measured protection and stopping for gradient-selected paired
neurons, gain/risk-selected neurons, random paired neurons, and official LoRA.
Freeze data, budgets, rates, maximum proposals/backtracks and success criteria
before submission. The completed calibration's supports may be reused only as
exposed development supports, with their discovery cost shown separately.

Judge complete held-out answers, genuine preservation errors versus aliases,
refusals, truncations, and total adaptation/export cost. Lower KL without
completed edits cannot win. Passing a development pilot would still require
fresh cases, additional parameter/data budgets and another task family before
recommending a recipe. The measured guard and KL constraints have prior art;
the useful contribution must be earned by their combined results.

```sh
PYTHONPATH=.:src python -m pytest -q tests/test_guarded_step.py tests/test_guarded_training.py
```
