# Score capture is not trained utility

23 October 2026. Offline audit of the completed relation-learning comparison;
no model execution, new fitting, or test-based support selection.

The saved Fisher selection is doing what its score asks. It captures 43.245%
of total empirical-Fisher group score across 350,208 MLP neurons using 173
neurons. Random seeds 43 and 71 capture 0.02096% and 0.02378%. Nevertheless,
their validation-selected trained task scores are 64/64, versus Fisher's 63/64.
The selected score mass is roughly 1,800–2,100 times larger; it is not a similarly
large advantage in useful adaptation. This does not prove a negative correlation
or explain the causal source of Fisher's one error.

Moving the same 173 slots freely across layers raises captured score to 44.190%.
The existing balanced allocation already attains 97.862% of this score-only
optimum. Thus layer allocation leaves little of this particular score uncaptured.
This does not prove that a global support would behave identically. It does rule
out claiming we failed to select large scores without inspecting the artifacts.

The reproducible [audit](../experiments/relation_learning/score_audit.py) reads
saved scores and supports, validates the recorded Fisher top-k selections, and
records [source hashes and results](../experiments/relation_learning/complete_v2/score_audit.json).
The regression test checks score mass against hand-computed arithmetic and
prevents accidentally squaring Fisher values twice.

## What the papers justify

[FISH, sections 2.1–2.2](https://proceedings.neurips.cc/paper/2021/file/cb2653f548f8709598e8b5156738cc51-Paper.pdf)
motivates its mask through sensitivity and evaluates the resulting training
empirically. Its empirical variant uses squared supervised gradients. That
quantity is a ranking heuristic, not a guarantee of an Adam update's benefit.
Our grouping and fixed layer quotas differ from its scalar mask.

[FISH-Tuning](https://arxiv.org/html/2504.04050v1) already explores applying
Fisher masks inside LoRA and other PEFT representations. Merely combining
selection with LoRA is therefore not an unexplored product differentiator.
These sources motivate baselines; neither establishes our quality advantage.

For a proposed first Adam update with zero moments and no weight decay,

\[
\Delta\theta_i=-\eta\frac{g_i}{|g_i|+\epsilon},\qquad
\Delta L_{\rm eval}\approx\nabla L_{\rm eval}^{\top}\Delta\theta.
\]

By contrast our stored selection score is an average of squared *individual
training-example* gradients. It loses signs and agreement between examples;
it also does not include the preservation gradient used in training. Neither
signed agreement nor actual Adam directions can be recovered from the saved
norms. Do not claim those factors explain this result without measuring them.
Clipping, subsequent moments and finite updates further limit the approximation.

## Experimental consequence

Do not launch a global-top-k replacement just to capture the remaining score.
Do not add a new selector based only on this diagnosis. First determine whether
the existing method has an actual low-data advantage, where the current 64-example
ceiling cannot hide one. That comparison must count selection examples as training
data, share preservation data, hold successful update count fixed, retain LoRA
and both random supports, and use validation choices plus fresh confirmation
inputs. Reducing data and parameters simultaneously would obscure the cause.

Prediction to test: with fewer task examples, all methods may still learn this
simple distinction; if so, close this synthetic benchmark as a method discriminator.
A useful positive result requires a selection advantage in complete answers at
the same data and parameter budget, without additional control damage and with
discovery overhead reported. It would still require another task and replication.
No new run or superiority claim follows from this offline audit alone.
