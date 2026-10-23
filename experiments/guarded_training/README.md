# Measured preservation during fitting

Experimental implementation. The [completed eight-fit pilot](complete_v1/REPORT.md)
fails its frozen advancement criteria: risk-aware selection, random and LoRA
each transfer3/8 answers. **No advantage over LoRA is established.** A saved-logit
audit also finds that its preservation rule rejects valid alias changes. The
replacement contract remains experimental; see the [current decision](../../docs/current-method.md).

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

## Completed pretrained comparison

The [frozen protocol](protocol_v1.json) fixes eight fits on Qwen3-4B, with
60 proposals and four backtrack scales at most. Each method uses the same
previously successful nominal learning rate,0.0003, full-batch objective and
0.9 complete-training-token confidence stop. Actual mean/worst-token
preservation KL limits are0.02/0.2. These are development design limits,
not universal safety thresholds. Model loading/worker execution has a3000-second
internal limit, the worker process3300seconds, and a requested provider session
limit5400seconds. Setup subprocesses separately have300/300/900-second limits;
none of these bounds queue wait.
The [accepted Kaggle version1](launch_v1.json) has byte-identical frozen source
and is COMPLETE. All eight fits and806 generated observations are independently
verified; the source and criteria remain fixed. The [report](complete_v1/REPORT.md)
records output review, failed criteria and the protection-contract diagnosis.

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

Both random and LoRA must complete the edits for the primary superiority
comparison to be informative. The candidate must complete8/8 merged transfer
answers without new factual preservation errors or truncation, retain its
measured preservation limits after merging, and lower held-out KL by at least20%
against each control in each case. Output review remains necessary even when
numeric conditions pass. Failure of a control is inconclusive, not an automatic
candidate win. Cached supports cannot establish measured whole-pipeline speed.

```sh
PYTHONPATH=.:src python -m pytest -q tests/test_guarded_step.py tests/test_guarded_training.py
```

## Saved-output contract diagnosis

`response_sets.py` is an experimental finite-response marginal likelihood helper,
not a semantic judge or a protected-training recipe. Its tests use independent
probability trees and check gradients, missing branches, duplicates, EOS and
FP16/BF16 accumulation. The saved-endpoint audit requires the private, already
downloaded research artifacts and the cached pinned tokenizer; it loads no model.
Use a new output filename because audit outputs are created exclusively.

```sh
PYTHONPATH=.:src python -m pytest -q tests/test_response_sets.py
PYTHONPATH=.:src python -m experiments.guarded_training.contract_audit \
  --parent artifacts/paired_mlp_v1/kaggle_download/paired-results/results \
  --results artifacts/guarded_training_v1/kaggle_download/guarded-results/results \
  --parent-proof artifacts/paired_mlp_v1/integrity_v1.json \
  --result-proof artifacts/guarded_training_v1/integrity_v1.json \
  --protocol experiments/guarded_training/protocol_v1.json \
  --output /tmp/neuronpatch-contract-audit.json
```

These conditional probabilities cover supplied finite strings, not every
semantically valid punctuation/case/wording variant accepted by a human or a
normalizing scorer. Unenumerated valid responses can still cause false alarms.
Known successful and harmful endpoint controls must qualify any next guard;
the complete-answer evaluation remains independent and necessary.
