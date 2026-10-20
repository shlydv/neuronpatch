# Sparse support and preservation must be enforced together

This is a source/math diagnostic, not a language-model result. It motivates an
implementation candidate and does not establish a novel theorem or priority.

## Audited implementations

- [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit/tree/b84624f44dfe8fc6cd9e41df916c44124a0c46dc),
  commit b84624f44dfe8fc6cd9e41df916c44124a0c46dc. Its evaluator builds an
  approximate null-space projector from activation-covariance singular vectors.
  The update solve puts the perturbation in the projector's range.
- [NMKE](https://github.com/LiuJinzhe-Keepgoing/NMKE/blob/6c81e45097c3e14d35360d852f9769da8abab81b/NMKE_main.py#L300-L306),
  commit 6c81e45097c3e14d35360d852f9769da8abab81b. It computes that solve, then
  multiplies rows by its neuron mask, before adapting to the actual weight layout.
  This masks down-projection input features after the protection operation.

NMKE's public snapshot contains partial integration files and no repository license
file was found. No source is vendored or executed here. Our independent matrix
diagnostic follows the operation order, not the complete attribution algorithm.
The paper already combines neuron masking with AlphaEdit: merely combining sparse
updates and a null-space projector would not be a new contribution.

## Why operation order matters

Let U contain orthonormal protected input directions and P=I-UUᵀ. The unmasked
update B satisfies UᵀB=0. A diagonal sparse mask M instead produces MB, and
UᵀMB need not be zero because M and P generally do not commute. Projecting again
can restore that constraint while adding nonzero entries outside the sparse mask.

For selected coordinates I, write B=E_I C, with E_I embedding the selected rows.
Find an orthonormal basis N for null(U_Iᵀ), then use B=E_I N Y. Both sparsity and
the specified protection constraint hold. Given edit keys K and requested
linear-output residual R, fit Y by regularized least squares:

    minimize ||K_Iᵀ N Y - R||²_F + λ ||Y||²_F

This also exposes available capacity: the number of free directions is
|I|-rank(U_Iᵀ). A zero-dimensional null space cannot implement a nonzero edit.
Finding no capacity should be reported, not hidden by applying another mask or
silently weakening preservation.

The intervention axis offers another route. For a PyTorch incoming projection
W[out,in], selecting output rows acts on the left, while protecting its input
directions acts on the right: ΔW=M_out D P_in. The two constraints are compatible
because they act on different axes. Cluster sharing on output rows is compatible
with the same right-side input projection. This makes NeST-style incoming updates
a sensible candidate to examine; it does not establish their behavioral advantage.
The diagnostic separately verifies this algebra on every fixture. It does not
compare edit-fit scores across different target output supports.

## Numerical check

[check_sparse_projection.py](check_sparse_projection.py) uses 16 seeded FP64 matrix
fixtures: 128 input features, 48 selected features, 12 protected directions, 4 edit
keys and 16 outputs. Its projected dense update uses the initial-edit form of the
audited solve with zero historical cache. All conditions use ridge 1e-4.

| Condition | Mean relative edit residual | Mean protected response change | Mean update norm |
|---|---:|---:|---:|
| Dense protective solve |9.07e-7|1.28e-16|0.776|
| Protective solve, then mask |0.640|0.110|0.472|
| Support-only refit |2.46e-6|0.374|1.262|
| Joint sparse/protected solve |3.39e-6|1.88e-16|1.463|

The joint solve retains exact support and protected responses in these fixtures,
at a larger update norm. An additional fixture with 8 selected features and 16
protected directions correctly has no free sparse direction and cannot fit the
requested edit. Raw values, script hash and runtime are in
[sparse_projection_checks.json](sparse_projection_checks.json).

~~~bash
python experiments/fine_reference/check_sparse_projection.py --output /tmp/projection-check.json
~~~

These fixtures do not show improved model behavior. They use random matrices;
actual activation ranks, conditioning, approximation thresholds, repeated columns,
deep nonlinear propagation and finite-precision merge behavior remain untested.
NMKE's empirical advantage might survive or benefit from its post-mask relaxation.
The source/math observation does not imply the paper's reported results are wrong.

## Candidate and next evidence

Prototype jointly constrained updates with an explicit capacity diagnostic.
First inspect actual activation geometry: a small FiNE support can have no
remaining direction after imposing protection. Incoming-row updates, as in
NeST's different intervention axis, offer a different constraint space; that
must be tested rather than assumed to solve the problem. Compare a joint solve
against dense protection, post-masking and a support-only refit using identical
keys, target residuals, supports and ridge settings. Record update norms.

A useful result requires actual generated edit transfer and held-out utility,
with matched parameter/data/compute budgets and fresh confirmation. Integrating
this geometric constraint into training or the public API comes after that
evidence. It is an attempt to address a concrete interaction between mechanisms,
not a reason to add every paper's technique to the library.
