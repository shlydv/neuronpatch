# Geometry of four verified FiNE update artifacts

This CPU audit inspects two successful GPT-J-6B source-parity edits and their
matching random-support controls. It does not train or change a model. The
[library parity report](../library_parity_v1/REPORT.md) establishes where the
stored updates came from. Exact artifact/manifest hashes and singular values
are in [summary.json](summary.json).

We stack each stored outgoing update as a row, preserving occurrence order,
then compute a float64 SVD. This is a **global basis across layers**, not a
factorization within each projection. Base output width: 4096. Case 1 has 20
occurrences but 19 unique selected columns; summing that duplicate is not a
substitute for actual FP16 sequential merge behavior.

| Case / support | Occurrences | Rank-1 squared energy | Rank-4 energy | Rank for 99% |
|---|---:|---:|---:|---:|
| 0, FiNE | 5 | 66.45% | 96.11% | 5 |
| 0, random | 5 | 32.39% | 89.95% | 5 |
| 1, FiNE | 20 | 62.72% | 82.77% | 16 |
| 1, random | 20 | 36.68% | 63.99% | 17 |

The selected updates share more direction than their random controls in these
two exposed cases. This is consistent with useful alignment, but two cases are
insufficient for a general claim. Rank 1 loses roughly one third of selected
update energy; rank 4 in case 1 loses 17%. BF16 factor reconstruction errors
are reported separately and are not generation errors.

Rank-4 factors use 80.10% of raw scalar storage for case 0 and 20.10% for case 1.
Retaining 99% of matrix energy needs ranks 5 and 16, leaving little storage
advantage. These ratios exclude support metadata and are not measured full-file
sizes or inference speedups.

**No compression win demonstrated.** Post-hoc reconstruction does not show
that fewer parameters can learn the same result, that decoded answers or utility
survive, or that training is faster. SVD, shared updates and low-rank adaptation
are established methods; no novelty is claimed.

A bounded follow-up, if this route is pursued, should decode fixed original,
rephrased and utility probes for full and rank-1/2/4/8 reconstructions and matching
random controls, reporting every setting and exact merge order. No-edit and
full-rank reconstruction controls must also run. Only behavioral preservation
would justify a shared factorization implementation, followed by a separate
direct-training test. This audit alone does not justify a new public API or
displace the correction/preservation objective.

```sh
python -m experiments.fine_reference.update_geometry \
  --source /path/to/four/parity-edit-artifacts \
  --output /tmp/fine-update-geometry
```

Source artifacts are our generated compact deltas, not full model weights.
The script checks four known artifacts; it is not a generic compression API.
