# Audit the adaptation mechanism before another selector

23 October 2026. The goal remains a useful adaptation method with a measured
advantage over strong baselines. This audit changes no frozen choices and launches
no GPU job. It examines twelve already trained sparse patches, including failed
high-rate fits, without loading model weights.

## What the papers actually contribute

[FiNE](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf)
combines localization with a specific editing objective and, for safety, staged
editing. Its claim cannot be reduced to selecting large gradients.
[NeST](https://arxiv.org/html/2602.16835v2) combines behavior probing with clustered
parameter sharing and supervised training. The existing source audit verifies
its shared-row operation, not its full empirical result. Our Fisher comparison
does not reproduce either complete method.

[LESS](https://pli.princeton.edu/blog/2024/using-less-data-tune-models) estimates
validation benefit using optimizer-aware training directions across warmup
checkpoints. It normalizes sequence gradients and uses projected LoRA gradients
for tractability. This is **data selection**, not a neuron-selection result.
Transferring its principle would require validating group-wise update predictions,
accounting for discovery cost and finite training trajectories. A first-step
gradient dot product is neither a new method nor a certificate of later benefit.

The primary abstract of the
[2026 targeted-selection study](https://proceedings.mlr.press/v306/nayak26a.html)
reports that gradient representations are useful but no selection algorithm
dominates, and gains depend on budget. Its full experiments and implementation
have not been audited here. It reinforces the need to separate representation,
selection, and budget rather than treating their combination as one explanation.

## New measurement: is there already a common update to recover?

For each layer and projection separately, form D with one selected neuron's
update per row. Transpose the outgoing-column tensor to this convention. No
cross-layer or gate/up/down correspondence is assumed.

- The least-squares identical-row approximation repeats the row mean. Its
  captured energy is k times the squared norm of that mean.
- The best rank-one approximation allows signed per-neuron coefficients. Its
  captured energy is the largest squared singular value of D.
- Normalize by the squared Frobenius norm of D. Aggregate by summed energies,
  not an unweighted average of percentages. Zero updates have undefined fractions.

These are post-training oracles in parameter space. They do not measure response
preservation, reproduce activation-profile clustering, or establish the effect
of training under either constraint. FP16 merged-weight rounding is not modeled.
Parameter-space geometry also depends on the model's coordinates.

| Saved condition | Task labels | Identical rows | Best rank one |
|---|---:|---:|---:|
| Validation-selected Fisher | 64 | 22.4% | 42.8% |
| Validation-selected random 43 | 64 | 21.4% | 40.9% |
| Validation-selected random 71 | 64 | 20.9% | 39.7% |
| Validation-selected Fisher | 8 | 21.4% | 44.3% |
| Validation-selected random 43 | 8 | 22.1% | 38.7% |
| Validation-selected random 71 | 8 | 20.3% | 43.2% |

All conditions also had twenty preservation labels. Across all twelve fits,
identical-row capture is 20.3–22.9%; rank-one capture is 37.7–44.9%. Each
projection has only four or five selected neurons. The raw spectrum is included
so future readers need not infer an effective rank from one summary number.

The [64-label audit](../experiments/relation_learning/complete_v2/update_geometry.json)
and [8-label audit](../experiments/relation_learning/complete_low_data_v1/update_geometry.json)
include source hashes, base fingerprints, every projection spectrum and all rates.
Three CPU tests independently check SVD reconstruction errors, opposing versus
identical rows, orthogonal rows, zero updates and invalid input.

## Decision

Do not implement a shared rank-one patch on the premise that our successful
updates already mostly have that form. They do not, in this metric. Do not claim
that this disproves clustered training: multiple clusters, broader supports and
training under a constraint are different experiments. Fisher's slightly larger
rank-one fraction is not better quality; its selected answers still lose to LoRA.

The next candidate must predict **transfer benefit of an actual update**, not
large task gradients, interpretability or post-hoc compressibility. Our earlier
functional-selection study already tried fresh-Adam gain and preservation risk
on two fact edits, so simply repeating that score is not progress.

Before a new full training comparison, a useful diagnostic must use a broader
behavior-learning development task, separate examples for training and transfer
calibration, and a finite update sequence with the same optimizer used to train.
Record predicted versus observed target and preservation changes for multiple
candidate supports, including random and LoRA. Check prediction quality before
using the predictor to choose a winner. Do not tune a score on these reviewed
relation tests or silently substitute an easy calibration task for useful
held-out improvement. Even a successful diagnostic would only justify a later
matched-cost confirmation experiment, not finish the library goal.

There is currently no evidence-backed new selection recipe to promote. Keeping
that conclusion explicit prevents another experimental heuristic from becoming
the library's advertised advantage.
