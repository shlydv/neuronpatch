# Broad adaptation fits the tiny training set, but does not generalize reliably

Completed 2026-10-19. Two learning rates, three checkpoints each, rank-2 LoRA on
all 72 gate/up/down projections in 24 layers. Each checkpoint has 829,440 trainable
parameters versus 43,008 for the row patches. The full table is in
[capacity/LEARNING_CURVE.md](capacity/LEARNING_CURVE.md). This is a larger-budget
learnability reference, not a parameter- or compute-matched comparison.

## Main finding

The higher-rate final checkpoint reaches **52/64 supported training exact matches
and 64/64 correct impossible refusals**, compared with the base's 20/64 and 17/64.
It retains 18/20 exact-correct training anchors. But on validation it gets only
**3/16 supported exact matches, 10/16 correct refusals**, and refuses 8/16 supported
questions. The lower-rate final checkpoint preserves all 20 anchors but gets only
4/16 supported validation exact matches and refuses 7/16 supported questions.

At the lower rate, validation supported EM goes 6→3→4 over epochs while training
supported EM goes 19→23→28. At the higher rate, training rises 24→31→52 while
validation goes 5→3→3. These curves support a learning/generalization gap in this
small-data setting. They do not prove ordinary statistical overfitting is the only
cause; article shift, task difficulty, objective design and label/scoring ambiguity
also remain plausible contributors.

The actual outputs still expose the intended failure: the final higher-rate model
answers RNA replicases for both enzyme presence and enzyme absence. It refuses the
correct Hoover question. More capacity has not produced reliable evidence support
checking on the unseen articles. Exact-match improvements on training can include
shorter wording and memorization, so they are not evidence of general reasoning.

## Computational correction

The first attempt retained full vocabulary logits for passage tokens that receive
no loss. It was deliberately stopped after profiling a 21.5 GB peak footprint and
a long Metal GPU wait on the 24 GB machine, before any epoch completed. Its code,
initial summary and log are preserved in capacity_memory_attempt/. It contributes
no scientific result.

The retry computes all hidden states but applies the vocabulary projection only to
the response-predicting suffix. It also releases unused MPS cached buffers between
steps. Full-versus-suffix causal CE, preservation KL and every parameter gradient
were checked on tiny Qwen on both CPU and MPS. The sampled retry peak through epoch
two was 8.4 GB; this is not a guaranteed peak over the whole run. The six successful
epochs took about 266 seconds of training, excluding model loading and evaluation.
Original and retry losses are numerically close, not bitwise identical. Exact
executed sources and hashes are retained. All six adapters were reloaded and
reproduced all 192 saved validation responses exactly; see capacity/reload_checks.json.

## Consequence for the combined method

The evidence now warrants a data-size learning curve before another localization
heuristic. Prepared nested training sets contain 128, 512 and 2048 examples with
unchanged article partitioning and full-context filtering. Expanded calibration
and validation each add 224 examples; old development examples are explicitly
separated from new ones. No predictions exist for the added examples yet.

The prior fixed three-layer controls were restricted references, not general LoRA.
Likewise, the paired score searched only those layers. Future discovery should
consider eligible layers globally and compare layer-matched random choices at an
explicit parameter/discovery budget. Larger-data and coverage effects must be
established separately before claiming the selector helps.

Confirmation remains unchanged and unevaluated. This experiment establishes a
stronger reference and a usable memory optimization, not superiority over FiNE,
NeST, SafeNexus or SpaRTA. The combined-method/library goal remains open.
