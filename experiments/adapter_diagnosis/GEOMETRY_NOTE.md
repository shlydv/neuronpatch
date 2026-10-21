# Weight geometry is a control, not a mechanism result

These post-freeze CPU measurements inspect the already selected parent factors.
They use no base model or new inference and change no factor, setting or group.
See [measured geometry](factor_geometry_v1.json) and the reproducible
[`geometry.py`](geometry.py) helper. The group-removal worker runs independently.

The squared Frobenius energy of the algebraic learned matrices is distributed:

| Projection family | Case 0 | Case 1 |
|---|---:|---:|
| Attention |44.0%|43.6%|
| Incoming MLP |46.2%|43.0%|
| Outgoing MLP |9.8%|13.4%|

The final three layers contain 12.1% / 14.4% of total energy; they overlap the
projection families and must not be added as a fourth disjoint category.
The groups also retain different numbers of factor scalars and projections.
A group-removal result cannot separate coverage, magnitude, co-adaptation and
representation with one scalar metric. No norm rescaling is applied in the
frozen diagnostic, and these numbers do not identify causal importance.

Within each MLP family, the learned weight update is not concentrated into a
few neuron-coordinate rows/columns. Across 458,752 coordinates per family,
the 32 largest account for approximately 0.10–0.12% of that family's energy.
Retaining 90% of outgoing-update energy would require 233,504 / 219,752 columns;
incoming-update energy requires 263,373 / 257,911 rows.

This is a statement about approximating the full learned **weight matrices**.
It does not establish how many neurons are needed to reproduce the behavior,
rule out a different compact solution learned from scratch, or rank functional
importance. Active feature distributions and downstream computation matter.
Taking the few largest weight changes is therefore not a justified shortcut
from a working LoRA patch to an equally effective neuron patch. Any compression
must be evaluated on correction and preservation behavior.

For Wdelta=B A, column energies are diag(A^T B^T B A), and row energies are
diag(B A A^T B^T). The implementation computes their diagonals in rank space
using float64 arithmetic on saved FP32 factors. These are algebraic norms,
not actual rounded FP16 merge differences. A separate tiny dense-matrix check
verifies both orientations; pinned and modern local runtimes agree on norm
totals and the reported energy-coverage thresholds. No full-size behavioral
conclusion follows from that numerical check.
