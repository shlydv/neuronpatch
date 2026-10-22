# The selector finds correction sensitivity, not protected edits

The [saved discovery-score audit](discovery_geometry.json) reads existing
gradient scores and fitted supports for both exposed development facts. There
are no new fits, model forwards or output evaluations. Reproduce with
`python -m experiments.paired_mlp.discovery_audit --help`.

Each paired neuron contains disjoint gate/up rows and a down column. The saved
score is the norm of that group's mean complete-correction-loss gradient over
three training wordings. All neurons have the same parameter cost in this
model. Summing squared group scores therefore gives the squared gradient norm
available on a chosen support. For vanilla SGD at an infinitesimal step this
determines first-order loss decrease. It does not determine Adam's actual step
or finite-step behavior.

| Recorded support | Case0: share of squared correction gradient | Case1: share |
|---|---:|---:|
| Guided, current near-uniform layer quotas |49.127%|49.391%|
| Random seed11 |0.0208%|0.0161%|
| Random seed29 |0.0235%|0.0203%|
| Random seed47 |0.0365%|0.0285%|
| Global top173, same score and cost; not trained |50.557%|51.710%|

All recorded supports contain173 of350,208 candidate intermediate neurons.
The audit requires their exact original indices and the guided per-layer
ranking to match. Allowing global allocation across layers instead increases
the score by only about3–5% relative to the current support. The existing
quotas retain97.17% and95.52% of the same-budget global score optimum.

This weakens the specific hypothesis that poor layer allocation leaves most of
our own score unused. It does not rule out behavioral differences from another
layer allocation, because this score ignores preservation and optimizer
geometry. Global supports have not been fitted and are not a new result.

The distinction matters: the guided selector concentrates the recorded local
correction gradient far more than these random controls, but the completed
comparison still shows no generated-answer advantage and genuine interference.
Thus correction sensitivity is an inadequate proxy for a useful, isolated edit.
The separate [weight-update audit](UPDATE_AUDIT.md) also shows that small weight
changes do not suffice to protect answers.

The existing checkpoint study can test one possible efficiency benefit: whether
guided supports reach an adequate complete answer sooner under the same stopping
procedure. It cannot turn this gradient-score gap into a quality or cost claim
by itself. A future selector still needs measured effects on preservation
examples, equal baseline treatment, and fresh confirmation before promotion.
