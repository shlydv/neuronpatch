# Complete post-update checkpoint study

**Accepted on Kaggle as version1; running at the latest check. No complete pretrained
checkpoint-study outcome yet.** See the [launch record](launch_v1.json) and
[Kaggle job](https://www.kaggle.com/code/shlydv/neuronpatch-paired-checkpoints-v1-20261022). This
tests whether excessive fixed-step fitting hides a useful earlier patch, while
separating an optimizer-step failure from a grad-enabled/evaluation-mode mismatch.
It follows the [completed live/merge replay](../paired_endpoint/complete_v1/REPORT.md),
which found all248 answer pairs identical across deployment states.

## Frozen comparison

Seven fits reuse the exact two exposed counterfactual facts, three correction
answers, six auxiliary/preservation answers, selected supports, optimizer,
cycling and comparable1.33million-scalar budgets from the original comparison:

- Two facts each: gradient-paired, clean random seed11, official rank-one MLP
  LoRA; all use lr0.0003.
- One additional diagnostic: case0's known failing gradient-paired lr0.001.
  It is reported separately and cannot become an extra tuned candidate.

Every six updates and at50, measure all correction and auxiliary response
tokens including EOS. Select the **first checked live checkpoint** where every
token has probability at least0.9. No validation or test answer chooses it.
Continue the trajectory to50 solely to recover the exact frozen reference.
Require identical final trainable tensors, all training records, live/merged
teacher logits and final generation controls. A failed replay prevents a causal
stopping claim. The objective and examples are unchanged, including the auxiliary
reference that initially differs from the base in each case.

The actual worker records full teacher logits at every check, then evaluates the
serialized/merged first-adequate and50-step endpoints on the same target, other
subject and utility test banks. At most434 scored generation observations are
produced; a fit with no adequate checkpoint has only its50-step endpoint. No
checkpoint is substituted merely because its test answers look better.

## Diagnose the final update

At selected and final checkpoints, compare grad-enabled and no-grad teacher
predictions at exactly the same weights. The high-rate diagnostic also records
the exact final sampled prompt before/after its final update, along with
pre-update parameters, clipped joint gradients, correction-only gradients,
Adam moments and resulting parameters. Independent checks reconstruct the Adam
step and directional scores. These are measured autograd quantities, not a
global nonlinear-preservation guarantee or attribution to a unique neuron.

The offline [step decomposition](step_analysis.py) further separates the recorded
Adam update into contributions from its current clipped gradient and previous
first moment, at the same recorded second-moment denominator. It reports their
dots with both the joint and correction-only gradients and retains the floating
point reconstruction residual. This can distinguish local objective conflict
from a historical contribution pointing uphill. It does not simulate removing
momentum or establish which alternative optimizer would improve model answers.
Synthetic tests recover actual PyTorch Adam steps in FP32 and FP64; pretrained
diagnostic tensors from the running study remain pending.

~~~sh
python -m experiments.paired_checkpoints.step_analysis \
  --results /absolute/path/to/checkpoint-results/results \
  --output /absolute/path/to/step-decomposition.json
~~~

## Count costs honestly

Discovery reruns once per fact and must reproduce the frozen support. Record
attachment, training, complete-response checks/raw-logit writes, serialization
and research diagnostics separately. The measured stopping prefix ends at the
selected checkpoint; continuing its control suffix is additional experiment
work. The fixed reference includes training plus one final check and export.
Neither number is a production deployment-throughput benchmark.

The original sparse fit/export/evaluation lifecycle took about230seconds per
low-rate fit, versus68–70seconds for LoRA; those totals do not isolate training.
Thus fewer optimizer updates alone cannot establish a useful cost advantage.
Nominal weight-update norms use direct sparse deltas or effective LoRA BA
products, never raw LoRA-factor norms; they exclude FP16 merge rounding.

Predictions fixed before inference are in the [protocol](protocol_v1.json).
The useful result requires complete transfer, no new unambiguous preservation
error on review, and a better correction/preservation/cost tradeoff than both
baselines under the same stopping procedure. All outputs and costs remain
reported even when that signal is absent. This is not a novel loss: stopping
and update-size control already appear in FiNE and other optimization methods.

## Checks and limitations

The actual seven-fit worker is exercised on tiny Qwen3 in FP32 and FP16, alongside
the original24-fit reference and eight-endpoint replay. Tests recover identical
reference parameters/history/logits, independently recompute checkpoint gates
and Adam updates, reject selecting a later qualifying checkpoint, and exercise
the no-qualifying-checkpoint path. A dense-product oracle checks effective LoRA
norms. These are numerical checks, not language-quality results.

The protocol limits the worker to6000seconds with a6300-second outer timeout;
dependency setup is additional. Full calibration tensors are retained, with
roughly1GB of expected private output. Base weights are not downloaded or run on
the laptop. Both parent datasets stay in their existing private Kaggle outputs.

Only two exposed development facts and one deliberately strong clean random
seed are used. Original exact-answer scores stay frozen; known Unicode, alias
and ambiguous-question defects are covered by qualitative review, not silently
regraded. Fresh confirmation remains untouched. This test cannot establish
paper superiority or a generally robust library recipe.
