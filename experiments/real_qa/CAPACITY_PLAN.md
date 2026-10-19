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
