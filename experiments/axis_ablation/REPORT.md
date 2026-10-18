# Editing the same neuron indices through different projections changes the trade-off

Exploratory ablation, 2026-10-19. Uses the already-exposed preservation_qa data.
This is mechanism-development evidence, not fresh confirmation or reproduction
of NeST, FiNE or SafeNexus. See [fixed protocol](PROTOCOL.md).

| Parameterization | KL | Supported correct / 48 | Correct abstentions / 48 |
|---|---:|---:|---:|
| Selected down columns | 0 | 41 | 46 |
| Selected down columns | 1 | 44 | 42 |
| Selected gate rows | 0 | 48 | 42 |
| Selected gate rows | 1 | 48 | 42 |
| Selected up rows | 0 | 48 | 43 |
| Selected up rows | 1 | 48 | 43 |
| Random down columns | 0 | 47 | 42 |
| Random down columns | 1 | 47 | 42 |
| Random gate rows | 0 | 48 | 37 |
| Random gate rows | 1 | 48 | 39 |
| Random up rows | 0 | 48 | 42 |
| Random up rows | 1 | 48 | 42 |
| LoRA down | 0 | 43 | 43 |
| LoRA down | 1 | 47 | 43 |
| LoRA gate | 0 | 47 | 43 |
| LoRA gate | 1 | 47 | 43 |
| LoRA up | 0 | 25 | 48 |
| LoRA up | 1 | 43 | 45 |

Base: 48 supported and 24 correct abstentions. Every setting retains 16/16 on the
exposed arithmetic sentinel. Direct patches have 43,008 parameters; each LoRA has
34,560. Each fit uses the same lr .0003, training order, data and 32-step budget.
These are fixed-rate comparisons, not independently optimized axes.

## Findings

For identical selected neuron indices, up/gate row updates preserve all supported
answers, whereas down-column updates break several. At KL=0, moving from down to
up recovers seven supported answers while giving up three correct abstentions.
Up rows dominate the KL=1 down-column point on both metrics in this experiment.
This supports treating intervention location as a separate design decision. It
does not establish that rows are inherently safer: effective learning-rate scale
and nonlinear sensitivity differ between projections.

The selected up support is only one correct abstention ahead of random up, with
identical supported accuracy. One seed and 48 examples per class cannot establish
a robust localization advantage. Selected gate beats random gate on abstention
here; that too needs independent confirmation and tuned controls.

Preservation KL strongly helps LoRA up, but leaves selected up/gate discrete
accuracy unchanged. This interaction matters: parameterization and objective
should be studied jointly. LoRA up with KL has more correct abstentions than
selected up but fewer supported answers, so there is no universal winner.

## Library and deployment changes

`NeuronPatch(..., axis="output")` now attaches compact selected-row deltas while
freezing base weights and biases. `axis="input"` retains the old column API.
All direct-patch evaluation was performed after save/reload; fixed-input logit
checks have zero error. A pretrained selected-up merge check is recorded in
`results/deployment.json`. Numerical tests cover dense-reference gradients,
unselected row invariance, base freezing, Qwen gate training, merging, artifact
reload and invalid-input atomicity. Legacy column artifacts remain loadable.

Exact model revision, versions, data/source hashes, raw predictions, training
losses and timings are in [results/summary.json](results/summary.json).
No model or patch binaries are committed. Timings exclude setup and evaluation.

## Next action

Advance up-row patches as a candidate, not a validated default. Use independently
split real-document data, multiple seeds and equally tuned LoRA/random baselines.
Compare correction/preservation frontiers instead of selecting a best aggregate
score. Reference preservation on real questions must distinguish already-correct
base answers from mistakes we want to fix. The official SQuAD corpus has been
prepared with provenance, but no model results on it have been generated yet.

Further synthesis still requires testing localization methods and shared capacity,
not simply calling this row-update implementation a new superior method. The
larger project goal remains unmet.
