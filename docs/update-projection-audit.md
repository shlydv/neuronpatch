# Can the existing sparse patch copy the LoRA update?

This CPU-only audit uses the completed boundary-transfer-v2 rank-one LoRA
endpoint and four saved support/group manifests. It loads adapter factors only,
not model weights, and makes no inference calls. The current Kaggle supervision
experiment is unchanged. The LoRA endpoint is not a successful boundary repair:
it has the full-document preservation failure reported in the authority study.

## Exact question and calculation

For each linear projection, the LoRA update is `D = scale * B @ A`.
For a fixed selected set of rows and disjoint groups, the nearest shared-row
update in squared Frobenius distance assigns each group the mean of its target
rows and assigns zero to unselected rows. This is an orthogonal projection.
Its squared residual is `||D||² - ||projected D||²`.

The implementation computes norms through `A @ A.T`, without materializing the
large dense updates. An additional oracle keeps the 32 largest row norms in
each permitted projection with independent row updates. This is an upper bound
on captured weight energy for any two-group, 32-row patch in those projections.
It is not a cost-matched trainable baseline or a selector available before
training LoRA.

## Results

All percentages below use the **entire LoRA update's squared Frobenius norm**
as denominator, except the final column.

| Support | Selected rows, independent | Selected rows, shared | Shared / independent |
|---|---:|---:|---:|
| Matched-authority | 0.2194% | 0.01424% | 6.492% |
| Old guided | 0.1985% | 0.01120% | 5.641% |
| Random43 | 0.2120% | 0.01169% | 5.515% |
| Random71 | 0.2112% | 0.01234% | 5.844% |

Allowing every row in gate/up projections retains 64.584% of the LoRA energy;
the remaining 35.416% lies in other projections. Even the oracle 32 independent
rows per gate/up projection retain only **1.9562%** of the entire update.
Thus most of this particular LoRA update lies outside the sparse support family,
before considering group sharing. The roughly 6% sharing retention is close to
the 2/32 dimension ratio; this alone is not evidence of defective clustering.

## Decision and limits

Do not pursue direct weight projection of this LoRA endpoint as a promising
compression shortcut. The exact least-squares calculation shows that this
representation cannot closely copy these particular weights. Activation-aware
distillation would ask a different question, incur additional training costs,
and first needs a teacher that actually passes preservation checks.

This does **not** establish a lower bound on achievable behavioral quality.
Different parameter updates can produce similar answers; weight coordinates
have different activation sensitivity; small discarded changes can matter more
than large retained changes. These percentages are not percent accuracy,
functional capacity, or causal contribution. They also do not explain why
random43 transfers better than random71. Do not select future supports by these
norms alone or present this diagnostic as a new adaptation method.

The useful distinction is between finding a different successful sparse update
and copying a dense one. The active fixed-support supervision experiment tests
the first route. Its behavioral results remain decisive.

## Reproducibility

[Machine-readable results](../experiments/instruction_boundary/complete_authority_v1/lora_projection.json)
include source hashes, per-module energies, and exact aggregate fractions.
[Implementation](../experiments/instruction_boundary/update_projection.py)
accepts `--lora DIRECTORY`, repeated `--patch LABEL=DIRECTORY`, and a new
`--output FILE`. It rejects unhandled tensor keys and nonstandard LoRA variants.
Two tests compare the low-rank computation against an explicit dense orthogonal
projection and check cancellation of oppositely signed rows under sharing.
