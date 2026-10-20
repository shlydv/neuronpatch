# Experimental protected incoming-row updates

This prototype tests a concrete interaction between the reviewed methods:
selected incoming-neuron rows and input-subspace protection act on different
axes, so one does not undo the other. The
[source/math note](../fine_reference/SPARSE_PROJECTION_NOTE.md) explains why
post-masking down-projection columns can instead violate a protection constraint.
Fixed input-subspace protection is existing work, not a novelty claim.

## Implemented

The caller supplies calibration projection inputs and selected output-row IDs.
The numerical input span is fitted with FP64 SVD, using the size-scaled
machine-precision rank threshold. A full span is rejected because no adaptation
direction remains. No statistical truncation silently weakens protection.

Each raw row D is used as D_eff=D-(D Q)Qᵀ with a fixed orthonormal protected basis
Q. All original weights are frozen. Scoped hooks remove themselves on errors;
the model must stay in eval mode. Allocated trainable parameters and effective
linear dimension are reported separately because the raw parameterization has
redundant directions. Export compiles the projected delta into an ordinary
NeuronPatch row artifact and invalidates the training wrapper.

The guarantee concerns the supplied input vectors in exact arithmetic. It does
not imply preservation on new prompts, cached-generation states or low-precision
export. Protecting a full model prefix requires all relevant unpadded token inputs,
not an arbitrarily selected final token. Mutable training buffers and concurrent
edits are not supported. The prototype currently uses CPU/CUDA FP64 arithmetic.

## Checked offline

~~~sh
python experiments/protected_rows/check.py --output /tmp/protected-row-check.json
~~~

A two-layer nonlinear FP64 network learns a representable change while its three
calibration anchors remain unchanged within 1.12e-16. MSE falls from 0.00487 to
0.0000189 on correction inputs and from 0.00674 to 0.000276 on fresh synthetic inputs.
Original weights are untouched; ordinary patch export/reload and merge match the
live prototype within 1e-12. Full-span rejection and exception cleanup are checked.
There are 96 allocated parameters and 75 effective linear dimensions.

These are mathematical/deployment integration checks, not LLM quality results.
Raw results and code hashes are in checks.json. This remains outside the public API.

## Next bounded model comparison

Freeze a capable instruction model, task data, calibration anchors, decoding,
layer/row budgets and tuning budget before fitting. Use baseline-correct
calibration answers to construct protected input spans; keep their articles and
all final utility/target evaluation inputs separate. Measure actual rank and
remaining capacity before training; report an infeasible configuration.

Compare the same selected rows with and without projection, projection with
layer/count-matched random rows, and a reasonably tuned LoRA baseline. Keep data
and tuning budgets equal, and report actual parameter counts, discovery cost,
projection cost and export size. Both row variants receive the same correction
and preservation examples. Rank selection by trainability in the permitted space
is a hypothesis requiring its own random-control comparison.

The useful outcome is better generated correction/abstention while retaining
already-correct held-out answers and unrelated utility. Inspect full responses,
not just teacher-forced averages. Evaluate the exported model as well as live
training. If this exploratory pilot shows a benefit, freeze the candidate and
confirm it on fresh data, several seeds and another model/task. Neither the tiny
network result nor source parity completes that goal.
