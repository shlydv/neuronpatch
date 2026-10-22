# Fixed directions control capacity; tiny trainable counts are not a storage result

22 October 2026. This is a research decision, not a new adapter or pretrained
result. The sixteen-edit Kaggle calibration remains frozen and pending.
The [source manifest](subspace_sources_v1.json) records inspection limits and
byte accounting. No additional GPU job or upstream implementation is run.

## Relevant existing mechanisms

- [LoRA-XS](https://arxiv.org/html/2405.17604v2), Section 3, fixes weight-derived
  projection matrices and trains an intermediate rank-by-rank matrix.
  The [inspected source](https://github.com/MohammadrezaBanaei/LoRA-XS/tree/e50b1a82b5d7b7e87a0c46bc62e07e54f6648d31)
  freezes A/B, creates the latent mapping, and uses B·R·A in both the live
  branch and merge calculation. Its initialization uses truncated SVD.
  This establishes an existing fixed-basis mechanism, not pretrained parity
  or verification of its checkpoint-storage implementation.
- [SVFT](https://arxiv.org/html/2405.19597v1), Sections 3.1–3.2, learns sparse
  coefficients over outer products of pretrained singular vectors, with
  diagonal, banded, random and top-k patterns. Learning scalar coefficients
  over fixed directions is already an established parameterization. Its
  code and reported benchmark results are not reproduced here.
- [FILet](https://arxiv.org/html/2605.01046v1), Section 3, prioritizes low-Fisher
  directions for LoRA initialization. Its [inspected selector and statistics](https://github.com/IKMLab/FILet---Learning-in-the-Fisher-Subspace-A-Guided-Initialization-for-LoRA-Fine-Tuning/tree/24d3d573b7921d5ee5c63ac773f170849c01992a)
  use input and fixed-label loss-gradient second moments, normalized
  weight-derived surrogate columns, and their projected-energy product.
  This differs from our conditional-softmax probes on a separate preservation
  bank, but rules out claiming low-Fisher direction selection itself as new.
  Neither its training wrapper nor empirical performance is verified here.
- [TinyLoRA](https://arxiv.org/html/2602.04118v1) reports very small updates for
  reasoning under RL, with a materially different capacity requirement under
  SFT. Its result is not evidence that173 coefficients can learn our edits
  or that changing an adapter can replace the learning objective.
- [Retaining by Doing](https://arxiv.org/html/2510.18874v3), introduction and
  Section2, motivates attention to training-data policy alignment. It explicitly
  distinguishes initial-policy Self-SFT from approximately on-policy refreshes.
  It does not establish that simply replacing our canonical reference paths
  with cached initial responses will fix interference. Our context audit is
  a coverage measurement, not its reproduction.

## What a fixed neuron basis would actually cost here

Suppose each selected paired neuron has a fixed, task-derived update direction
and only its coefficient is trained:

\[
\Delta\theta=\sum_{i=1}^{173}c_i d_i.
\]

This reduces trainable degrees of freedom from 1,328,640 to 173. However, each
direction contains two incoming rows and one outgoing column, each of width 2560.
Storing all directions in FP32 requires 5,314,560 bytes; the coefficients add 692.
The combined raw tensors therefore require 5,315,252 bytes before metadata and
container overhead. This is dimension-based accounting, not a measured adapter
file or runtime benchmark. Sharing, regenerating or compressing those
task-specific directions would require an explicit, verified export contract.

The possible benefit is optimization-state reduction and a capacity constraint.
There is no automatic reduction in deployed patch size, no guarantee of safer
answers, and no new fixed-basis principle. Any later test must compare an
adequately tuned fixed-basis control and expose basis/discovery costs, alongside
ordinary LoRA and random neuron supports. Do not launch that branch merely
because a tiny coefficient count looks attractive. First resolve the current
finite-update calibration and inspect all complete outputs.
