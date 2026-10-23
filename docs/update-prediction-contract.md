# Predict the update that training actually takes

23 October 2026. This is experimental measurement infrastructure, not a new
selector, a protected training algorithm, or evidence of improved model answers.

## Why this differs from the earlier diagnostic

The functional-selection experiment constructed a unit-normalized fresh-Adam
direction from mean correction gradients. It did not model later momentum,
the combined task/preservation gradient, global clipping or a multi-step fit.
The current training loop uses all of these except direction normalization.
Reusing the first-step proxy would not test whether the actual optimizer's
updates transfer to separate examples.

The [LESS implementation read on 23 October](https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/collect_grad_reps.py)
constructs a moment-aware feature in `obtain_gradients_with_adam`: updated first
moment divided by the square root of updated second moment plus epsilon. It
does not execute our optimizer step. That feature is useful precedent for
influence estimation, not an exact displacement oracle for this trainer. This
was a read of the current source, not a pinned reproduction or a claim that LESS
is wrong. No source was copied.

## Implemented observer

`experiments/relation_learning/update_prediction.py` previews the next ordinary
Adam/AdamW update on independent compact parameters, gradients and optimizer
state, using PyTorch's actual optimizer. A scalar cap prevents accidentally
cloning a full model. Parameters with no gradients preserve their optimizer skip
semantics. The real optimizer is neither stepped nor changed during the preview.

For a separate calibration loss L at current parameters theta, the forecast is

    predicted change = gradient L(theta) dot (theta_next - theta).

Negative means improvement. Derivatives use `autograd.grad`, leaving accumulated
training gradients untouched. After the real step, the observer checks exact
parameter equality against the forecast and records observed loss change and
the linearization error. A state/gradient change between preview and execution
must not be treated as a valid measurement; mismatched parameters are rejected.

The preview must run after unscaling and clipping. The instrumentation is for
deterministic forwards and ordinary Adam/AdamW without hooks or differentiable/
capturable execution. The caller must preserve model buffers and RNG. The
retry-enabled training mode now permits read-only before/after-update observers
outside the retry loop. Callbacks inside retries remain prohibited. Two additional
FP16 overflow tests verify once-per-accepted-update observation, bit-identical
training, and no observer invocation when retries are exhausted. No GPU run was
submitted by this integration.

## Verified properties and limits

Six CPU tests cover both optimizers, multiple parameter groups, weight decay,
AMSGrad, absent gradients, later-step momentum and a cap/stale-state rejection.
The actual reference training loop is bit-identical with and without the observer,
including histories. Independent quadratic losses verify the forecast remainder.
A momentum fixture reverses the sign expected from a fresh gradient; a large-step
fixture has predicted improvement but observed harm. These are mathematical
controls, not language-model findings.

The test proves the measurement can distinguish optimizer mismatch from a local
approximation failure. It does not show that either explains our previous model
errors. Exact parameter previews can still have inaccurate loss predictions.
Loss prediction can be accurate while generated-answer predictions are poor.

Recomputing calibration gradients at each step measures local predictability
along a trajectory; it is not a prospective prediction of the entire trajectory
from its starting point. Keep these claims separate in the next protocol.
Calibration examples are development data and their compute/labels count toward
the method's budget. Final confirmation examples must never enter this observer.

## Next integration requirement

The24 October diagnostic `experiments/relation_learning/gradient_agreement.py`
reconstructs only fixed candidate incoming rows/outgoing columns from the existing
full-response gradient factors. Separate streaming banks retain signed means and
mean squared norms for training, calibration and controls. Their ratio
`||mean gradient||² / mean ||gradient||²` measures cancellation; it is not mean
pairwise cosine or a neuron importance score. Zero-gradient coherence is undefined.
Supplied optimizer displacements can be dotted into each bank without replacing
Adam by a gradient-descent assumption. This is for independent paired parameters;
shared cluster parameters require summing member gradients first.

Two added CPU tests reconstruct gradients against dense neural-network autograd,
compare signed predictions with central finite differences, and demonstrate both
Fisher-hidden cancellation and coherent updates that harm controls. Together with
the existing selection and actual-update tests, eleven tests pass. No real-model
gradient-agreement result or default selection change follows. Candidate IDs and
coordinate ordering must be identical across banks; separate example membership
and discovery costs remain caller obligations.

Use this observer to compare frozen candidate supports and LoRA on the same
development examples and actual optimization sequence. Record forecasts before
each applied update; do not use measurements to silently change supports, rates
or stopping. Include generation checks alongside losses at prespecified points.
An observer that accurately measures an inferior method does not complete the
goal. A future selection rule must earn its calibration cost through better
quality/preservation/cost on separate confirmation tasks.
