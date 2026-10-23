# Scope is a separate problem from sparse capacity

23 October 2026. This is a design assessment, not a new implemented method.
The [retrieval qualification](../experiments/retrieval_evidence/complete_v1/REPORT.md)
closed without training. The [precision/prompt diagnostic](../experiments/inference_sanity/complete_v1/REPORT.md)
also completed, with identical output tokens across its three numerical settings. We should not
interpret either as evidence that our neuron selector works.

## What the accumulated experiments actually constrain

Our two-fact comparisons show that several parameterizations can fit the desired
answers. Selecting a small set of coordinates does not by itself isolate the
semantic situations in which their update should matter. The earlier function
vector results and later collateral changes are compatible with this limitation,
but do not identify one shared causal mechanism. Conversely, the current retrieval
slice is largely solved already, so it cannot decide which parameterization is
best at learning an evidence-use skill.

There are two distinct product requirements: learning a requested change, and
applying it only where appropriate. A permanently merged sparse patch attempts
to satisfy both through the same weight update. This is a demanding requirement,
not something guaranteed by a low parameter count. An explicit conditional patch
could separate them, but would change deployment costs and introduce routing
errors. It must not be promoted as our invention or a free solution.

## Prior art that prevents an easy novelty claim

[SERAC](https://arxiv.org/html/2206.06520) separates an edit-scope classifier from
a counterfactual predictor and otherwise falls back to the base model. Its
QA-hard evaluation deliberately includes related but out-of-scope inputs.
This is direct precedent for treating where an edit applies as a learned problem,
not a consequence of sparse coordinates. We read the scoping/model sections;
we have not reproduced its experiments or inspected its full implementation.

[GRACE](https://arxiv.org/abs/2211.11031) uses a codebook of activation keys,
trainable values and deferral radii. Its
[official repository](https://github.com/thartvigsen/grace) is an appropriate
reference for conditional latent interventions. We inspected the project source
listing and paper description, not the underlying editor code successfully;
there is no source-parity claim here. A learned sparse patch plus a gate would
need comparison against such conditional adaptation, not only always-on LoRA.

The [INLAY scope study](https://arxiv.org/html/2608.26292v1) reports that its
particular benchmark queries gave an oracle router no advantage over a fixed
policy, then adds negatives to expose routing value. Its own audit also reports
a generation-path gate bypass. These are author-reported findings, not an
independent reproduction. They warn us to measure actual complete-output outcomes
for every candidate action and include genuine cases where the patch should not
apply. Their benchmark conclusion should not be generalized to every existing
editing evaluation: SERAC itself explicitly distinguishes hard scope negatives.

## What would justify implementation

After the inference check, freeze one application contract and distinguish its
in-scope corrections, hard neighboring out-of-scope inputs and unrelated utility.
First measure the outcomes of base, ordinary LoRA and sparse candidates on those
same development inputs. This gives an empirical upper bound on what choosing
between their outputs could achieve. If even the best available action fails,
a router cannot solve the problem. If always-on LoRA already dominates, routing
or neuron selection lacks measured value on that set.

Only if correction and collateral outcomes differ meaningfully should we test
conditional deployment. The comparison must include a task instruction, a simple
external relevance gate and gated LoRA with the same routing supervision. Benefits
shared by gated LoRA belong to routing, not sparse selection. The gate must execute
on the real generation path, with explicit handling of KV cache and multi-turn
scope. Frozen unseen examples must test both mistaken activation and mistaken
deferral. Memory, latency and adaptation cost include the router.

This is an alternative mechanism worth evaluating if justified, not authorization
to proliferate another framework now. The final goal remains a demonstrated
useful advantage over strong methods, including ordinary SFT/LoRA and relevant
paper baselines. Merely adding a router or constructing a favorable toy benchmark
would not meet it.
