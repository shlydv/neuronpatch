# Next reference: can broader adaptation learn the distinction?

The controls so far restrict even LoRA to up projections in three chosen layers.
That is a useful matched-small-update control, but not a strong general LoRA
baseline. After replay, protected training examples still fail. Before spending
more discovery effort, establish whether broader adaptation can learn the intended
behavior on these data. This is a capacity/optimization reference, not a novel
method, faithful paper reproduction, or parameter-matched superiority comparison.

Use the same pinned model and article splits. Train rank-2 LoRA on gate/up/down
projections in all 24 layers, with alpha 2, no dropout and bias frozen. Report actual
trainable parameter count (expected 829,440 versus 43,008 for row patches). Use the
same two learning rates, .0001 and .0003, up to three epochs, and save/evaluate after
each epoch. Supervised examples and replay anchors remain unchanged; replay KL
coefficient stays 1. Repeat the deterministic training order each epoch. Record
training cost and do not call a larger/more-trained baseline a fair budget match.

Evaluate training anchors, supported answers, correct refusals and validation at
each checkpoint. Inspect raw changes and retain all checkpoints, rather than using
only a best-looking aggregate. Confirmation stays untouched. If broader adaptation
learns the distinction, study its layer/update contributions and try compressing
useful changes into selected neurons. If it also mostly increases refusal, revisit
training targets and objective, and examine a larger model; do not claim our sparse
selection failed solely because of its capacity. A successful broad reference is
necessary evidence for this route, not fulfillment of the library goal.

Implementation: `run.py --capacity`. To limit activation memory for all-layer
training, supervised CE and replay KL gradients accumulate through separate backward
calls before one common clipping/optimizer step. A numerical test checks equality
to the joint objective. Each epoch records its own training duration, excluding
evaluation; the optimizer persists across epochs. The anchor stream continues
across epochs while supervised order repeats. Code/source hashes are retained.

Memory correction: the initial full-logit attempt was stopped after a measured
21.5 GB peak on the 24 GB machine, before any epoch completed. Its source and log
remain in capacity_memory_attempt/. The retry computes vocabulary logits only for
the supervised response suffix, including the last prefix position needed for causal
prediction. Full hidden-state processing is unchanged. Tiny-Qwen tests compare full
and suffix CE, KL and all parameter gradients. Unused MPS cache is released between
steps, and driver memory is logged. This is a computational implementation change,
not a reduced sequence length, reduced model, or altered training target.
