# Research provenance and implementation decisions

Updated 2026-10-22. This repository contains selected-neuron update primitives
and an experimental GPT-J FiNE-K workflow. The upstream FiNE editor has also been
run separately on a fixed 16-case 6B-model comparison, which completed.
These are bounded reference checks, not full-paper reproductions. Research source
is not vendored into the library.

For the completed 22 October comparison and current decisions, start with
[current-method.md](current-method.md). The source audits below distinguish
published methods from our experimental adaptations.

| Work | Reference and source | What informs neuronpatch | Status |
|---|---|---|---|
| FiNE | [paper](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf), [official code](https://github.com/opanhw/fine-unified-editing-code) | Per-token contribution selection, occurrence deltas, reference objective and stopping | CPU and bounded two-case 6B GPU workflow parity; separate 16-case upstream result. Full benchmark, safety and sequential editing not reproduced |
| NeST | [v2 paper](https://arxiv.org/html/2602.16835v2), [official archive](https://anonymous.4open.science/r/nestSP27-1C20/) | Probe-based selection, neuron clustering and shared trainable updates | Source inspected; actual shared-row forward/gradient/merge operation checked on CPU. Full discovery/safety results not reproduced; see [source audit](../experiments/nest_source/README.md) for precision and code/paper distinctions |
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

## Structured sparse fine-tuning audit (22 October)

[SIFT](https://github.com/song-wx/SIFT), source
`5bef8a06ed9fa2587013c6f898222d047bde56e2`, is additional prior art for compact
sparse weight increments and gradient-driven scalar selection. We inspected
`sift/sift.py` and package setup; no reproduction or speed/memory claim is
verified. Its indexed-gradient and hook lifecycle need numerical validation
before using it as an execution baseline. We did not vendor its source, and
absence of a root license in the inspected tree is not permission to reuse it.

[Pruning-based sparse fine-tuning](https://github.com/CenjhihLi/sparsity_finetuning),
source `bf38656696ad9e122edb319391d3a5325ce2b94a`, already supplies compact
input-column and output-row increments. The inspected
[`LinearFinetuner`](https://github.com/CenjhihLi/sparsity_finetuning/blob/bf38656696ad9e122edb319391d3a5325ce2b94a/methods/tp_finetuner.py)
constructs effective weights before its linear call. The training entry point
offers dependency-aware channel selection and several importance criteria;
the importance code aggregates groups of dependent operations. Exact gated-MLP
coupling and pretrained outcomes have not been reproduced here. The repository
has an MIT license; no implementation is incorporated into neuronpatch.
These findings further rule out claiming row/column sparse increments or
dependency-aware channel selection as our invention. The weight-space forward
also makes training/deployment arithmetic a relevant implementation comparison,
not proof that our pending merge discrepancy has any particular cause.

Our current synthesis must therefore be judged by measured selective adaptation,
cost and usable deployment, rather than by attaching a new name to sparse rows,
columns, gradient selection or ordinary stopping/regularization.

## Selection and retention controls checked on 22 October

[Ploner and Akbik, EACL 2024](https://aclanthology.org/2024.findings-eacl.122.pdf),
sections3.1–3.3, already compare downstream squared-gradient selection, a ratio
of downstream to pretraining squared gradients, random subsets and LoRA. Their
RoBERTa classification/NER experiments independently tune learning rates and find
competitive random subsets; the combined score does not significantly beat
random in their aggregate comparison. This is not an instruction-LLM neuron
editing experiment. It rules out presenting a correction/preservation gradient
ratio as our invention or assuming it will solve the observed interference.
Their linked code has not been inspected here.

[LS-LoRA, arXiv2610.11620v1](https://arxiv.org/html/2610.11620v1), sections3–4,
uses forward-only layer input/output cosine similarity for selecting adapter
locations. It reports adaptation and commonsense retention, including a control
with the same number of randomly selected layers. All methods use the same
LoRA rank, so its all-layer baseline has more trainable parameters. The paper
provides relevant evidence for inexpensive placement controls, not proof that
our neuron score works or that its proxy predicts preservation on our inputs.
Its empirical claims and implementation have not been independently reproduced.

## Fixed-subspace boundary checked on 22 October

The [fixed-subspace decision](../experiments/functional_selection/SUBSPACE_DECISION.md)
records additional relevant prior art and pins the inspected LoRA-XS and FILet
core source. It also separates trainable coefficient count from the bytes needed
for task-specific direction vectors. These findings constrain later claims and
baseline choices; they do not establish another successful method. The pending
functional calibration is unchanged.

## Validation-guided parameter selection checked on24 October

[DualSFT](https://arxiv.org/html/2605.06166v1), sections3–5 and appendicesB/F.10,
already scores parameters and data using a shared validation-improvement
approximation. It uses warmup, signed gradient interactions, a practical
diagonal correction and a preservation direction. The authors explicitly
distinguish their SGD-based local derivation from the AdamW-derived RMS-gradient
proxy; that proxy is not an exact Hessian. They test predicted rankings against
finite restricted-training outcomes at several horizons. The published setup
uses four80GB A100s and a5% parameter budget. No official implementation was
located in this inspection; no empirical result has been reproduced here.

This is direct prior art for choosing weights by predicted validation benefit,
not just an analogy from data selection. We must not present that idea as new.
Its individual-coordinate score also cannot be assumed correct for our shared
row deltas: sharing changes the update space and introduces cross-row effects.
Our actual-optimizer observer is the appropriate existing measurement tool;
the open issue is predictive utility across training/transfer examples, not the
absence of another scoring formula. No new GPU run follows from this review.
# Instruction-boundary objective prior art — 24 October 2026

Read while the fixed-support supervision experiment was already submitted; this
does not change its frozen inputs, objective, criteria or evaluation.

[StruQ](https://www.usenix.org/conference/usenixsecurity25/presentation/chen-sizhe)
combines an input-formatting front end with training that distinguishes the
instruction and data portions. It augments instruction tuning with instructions
inside data. Our ordinary JSON-quotation prompts and small adaptation are not a
reproduction of that system, nor a new invention of instruction/data separation.

[SecAlign, sections 3–4](https://arxiv.org/html/2410.05451v3) argues that training
only the desired response leaves the competing injected response underspecified.
It trains preferences between legitimate and injected responses and evaluates
security plus utility. Its claims are specific to its models, datasets and attack
evaluations; they do not establish that our fixed-support SFT repair will work.
This is direct objective-design prior art, not evidence for a particular sparse
selector. We read the paper, not reproduced its reported outcomes.

In the [official implementation](https://github.com/facebookresearch/SecAlign/blob/main/align.py),
`generate_preference_data` pairs an injected prompt with the original example's
response as chosen and the injected example's response as rejected. The inspected
entry point uses TRL preference trainers and rank-64 LoRA on q/v projections.
This differs from our rank-one all-projection LoRA and shared gate/up row patches.
Consequently importing its name or preference loss would not reproduce its
experimental configuration. No source code was copied into neuronpatch.

The [Meta SecAlign entry point](https://github.com/facebookresearch/Meta_SecAlign/blob/main/secalign_plus_plus.py)
defaults to self-generated responses and randomized injection positions and
launches a distributed LoRA DPO recipe. These are configuration observations,
not an audit of every helper or proof of their isolated causal effects.

**Inference for this project:** the pending run isolates missing trusted-context
supervision. A later preference-objective comparison would be a separate question
and should hold its data/support fixed. Do not add DPO, randomize positions, and
change selection simultaneously, then attribute a gain to neurons. If sparse
selection becomes useful, demonstrate that advantage against the same competent
task objective, not against a deliberately weak adaptation baseline. Passing
our small table task would still not certify general prompt-injection security.

