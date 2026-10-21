# Research provenance and implementation decisions

Updated 2026-10-20. This repository contains selected-neuron update primitives
and an experimental GPT-J FiNE-K workflow. The upstream FiNE editor has also been
run separately on a fixed 16-case 6B-model comparison, which completed.
These are bounded reference checks, not full-paper reproductions. Research source
is not vendored into the library.

| Work | Reference and source | What informs neuronpatch | Status |
|---|---|---|---|
| FiNE | [paper](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf), [official code](https://github.com/opanhw/fine-unified-editing-code) | Per-token contribution selection, occurrence deltas, reference objective and stopping | CPU and bounded two-case 6B GPU workflow parity; separate 16-case upstream result. Full benchmark, safety and sequential editing not reproduced |
| NeST | [v2 paper](https://arxiv.org/html/2602.16835v2) | Probe-based selection, neuron clustering and shared trainable updates | Planned; no official implementation verified in this review |
| SafeNexus | [paper](https://arxiv.org/html/2607.28969v1) | Selected-neuron masks on LoRA updates; layer-matched random controls | Planned; no official implementation verified. The author's Code link inspected during review led to a different project |
| SpaRTA | [paper](https://arxiv.org/abs/2502.15975), [code](https://github.com/IBM/SpaRTA), [package](https://pypi.org/project/peft-sparta/) | Random scalar-weight baseline and compact delta artifacts | Reference, not implemented; our random selector samples whole columns, not individual weights |
| ReFT | [paper](https://arxiv.org/abs/2404.03592), [pyreft](https://github.com/stanfordnlp/pyreft) | Trainable representation interventions | Adjacent approach; not the deployment mechanism in v0.1 |

Source versions inspected: FiNE `a2489cbac0dfe5a956385734065cfa041384e7d1`;
IBM SpaRTA `e3b91c598090a306b68005ff3f9fbba1eed6c84a`.
FiNE's code is MIT licensed and SpaRTA's code Apache-2.0; retain their notices if
code is incorporated later. Paper descriptions do not imply code availability.

## Why columns first

FiNE's `knowledge-editing/easyeditor/models/fine/fine_main.py` trains compact
vectors using selected inputs to `down_proj`, then adds them to the corresponding
weight columns. The generic API uses a PyTorch module; the occurrence-preserving
reference workflow uses scoped hooks on ordinary projections.
The base is frozen; only the compact delta is an optimizer parameter. This avoids
allocating a dense trainable matrix and relying on gradient masking alone.

A neuron in a gated MLP is not necessarily the output of `act_fn` alone. For
future selection, record `down_proj` inputs (after activation and gating), and
specify the exact quantity being ranked. FiNE, NeST and SafeNexus use different
selection and parameterization choices; they must not be conflated.

## Evaluation obligations

- Separate target improvement from collateral damage and over-abstention.
- Match random neuron counts **within each layer**, use multiple random seeds,
  and report actual scalar parameter budgets.
- Give LoRA a reasonable validation/tuning budget; publish its configuration.
- Include neuron discovery cost, training cost, and deployment checks.
- Test new document/topic families and prompt variants; never tune on final tests.
- Verify serialized and merged models, not just an in-memory training wrapper.
- Do not call activation correlations causal circuit discovery.

The CPU toy task deliberately gives the selector the true edited column. It tests
optimization and artifact correctness, not discovery quality or LLM generalization.

## Initial guided selector (implementation update)

The initial selector ranks `abs(mean_positive - mean_negative)` of actual
projection inputs. Means are computed over explicitly selected tokens within each
example, then equally over examples. This deliberately simple association
heuristic is **not** a reproduction of FiNE, NeST, or SafeNexus's localization.
The tiny Qwen3 test validates architecture compatibility and optimization only.
Its calibration groups are synthetic token distributions; no semantic meaning
or causal role is assigned to the resulting selected indices.

## Mechanism audit (2026-10-19)

See [method-design.md](method-design.md) for the intervention-axis distinctions,
FiNE paper/code KL discrepancy, located NeST archive, synthesis hypotheses and
requirements for a defensible improvement claim. NeST and SafeNexus primarily
modify gate/up output neurons; the current down-column primitive is not equivalent.
The archive could not be retrieved in this audit; code verification remains open.

## Preservation geometry audit (2026-10-20)

[AlphaEdit](https://arxiv.org/abs/2410.02355) projects updates away from protected
activation directions. Its code was inspected at
b84624f44dfe8fc6cd9e41df916c44124a0c46dc.
[NMKE](https://arxiv.org/abs/2510.22139) already combines neuron attribution and
dynamic sparse masks with AlphaEdit. Its partial source was inspected at
6c81e45097c3e14d35360d852f9769da8abab81b; no license file was found, and no source
is incorporated. The inspected operation applies its sparse mask after the
projected solve. Our [independent diagnostic](../experiments/fine_reference/SPARSE_PROJECTION_NOTE.md)
shows why post-masking can violate that linear constraint and checks a joint
solve, including an infeasible sparse support. This is not a model reproduction.

[Null-Basis LoRA](https://arxiv.org/abs/2609.25618) also studies fixed
activation-derived null-space parameterization throughout fine-tuning. Its paper
was inspected as adjacent work; code has not been verified. Thus fixed protected
input bases are existing ideas. The implementation candidate here concerns
compatible neuron-update axes, explicit remaining capacity, and measured
transfer/preservation; novelty and superiority remain unestablished.

## Additional neighboring work checked on 2026-10-21

[SCoNE](https://arxiv.org/html/2609.00689v1) combines positive attribution with
cross-input attribution variability, then statically scales selected down-projection
columns. This is training-free RAG adaptation, distinct from learning our sparse
deltas. The paper-linked implementation was inspected at
`3b02f40eefa8e878db7c5f88849365b25359c5ad`; no upstream code is incorporated.
A [CPU path-gradient audit](../experiments/attribution_audit/SCONE_PATH_NOTE.md)
reproduces a formula/implementation mismatch, including a constructed ranking
reversal. Its real-model effect is unmeasured; no benchmark reproduction or
superiority claim follows. Cross-input variability alone is not a new idea here.

[Faithfulness to Refusal](https://arxiv.org/html/2607.05355v1) already compares
selectors through row ablation, contrastive behavior signals and layer-matched
controls. Its operational tests concern sufficient sets, not a unique or necessary
circuit. Paper sections 1–3 were inspected; its results are not independently
reproduced. Stable rankings or transferable edits alone cannot establish a unique
causal mechanism in our work either.

[MicroEdit](https://aclanthology.org/2025.emnlp-main.1719/) uses sparse-autoencoder
representations for neuron-level knowledge disentanglement. The primary abstract
was inspected; code and full method remain unverified here. Polysemanticity and
editing spillover are existing research problems, not discoveries unique to us.
