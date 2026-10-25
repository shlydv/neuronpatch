# Distribution-retention control

Status: submitted; Kaggle reports RUNNING on 25 October 2026.

The [loss-isolation result](../experiments/instruction_boundary/complete_loss_isolation_v1/REPORT.md)
shows that both short-answer task training and short-answer preservation training
can reproduce the same clean failures. The next control changes the preservation
objective while holding its token trajectories fixed. This avoids attributing a
simultaneous change in answer style to the KL objective.

## First comparison

Keep the 186 diverse SFT task examples, rank-one all-projection adapter, initial
seed, Adam schedule, learning rate and 31 clean examples unchanged. Replace clean
answer NLL with mean token KL(reference || edited), initially with weight 1.
Use the exact same clean prompt and source-reviewed answer tokens, including EOS,
and the same seeded clean-example order. This is a standard regularizer, not a
novel method. Weight 1 is a bounded control, not a claim of optimal tuning.

Cache base logits at those prediction positions before attaching any adapter.
The exact tokenizer/config calculation gives 119 prediction positions across
31 responses, vocabulary 151,936, FP16 cache 36,160,768 bytes; the longest response
cache is 3,038,720 bytes. A 64 MiB host-cache cap leaves explicit headroom without
quantization or truncation. Only one response reference moves to GPU per update.
Actual forward/backward peak memory still needs to be measured on the T4.

The existing `preservation_kl` primitive supplies the loss. The experiment-local
cache binds the base and tokenized trajectory identities. The shared optimizer
records `preserve_kl` separately from `preserve_nll`; the unused field is null.
Fifteen focused CPU tests pass, including direct Adam agreement for KL plus SFT,
unchanged default behavior, ablation lifecycle and half-precision worker checks.

## What this comparison can establish

It tests whether matching probabilities on the *same forced contexts* produces a
better training tradeoff than reinforcing the hard reference tokens. It does not
test complete base-response distillation: the base may normally produce different
prefixes. A subsequent base-generated-trajectory comparison must be separately
identified if needed, rather than quietly changing this control.

Evaluate complete generated answers, paired clean regressions, legitimate task
changes, attack resistance, lengths and costs. Retain every failure. The old
30-document set is exposed development data; no result on it establishes fresh
transfer or qualification. A promising repair must proceed to unseen inputs and
unrelated-task retention before comparing guided/random neuron supports with a
competent LoRA baseline. The original library advantage remains unproven.

Before submission: integrate cache preparation and hashing into the runner,
measure initial reference KL and finite gradients, replay the existing combined
adapter, and verify export/reload and evaluation in the actual tiny-model worker.
Do not add new neuron selection, target wording, checkpoint selection or multiple
rates to this objective control.

## Integrated worker and frozen control

The shared runner now prepares and saves reference logits before attaching LoRA,
records their SHA256, tokenized-sample identities, base identity, bytes and cost,
and checks initial KL for every cached response. It rejects initial drift over
1e-6. The diagnostic worker replays the previous combined adapter, then performs
one KL+SFT fit and evaluates the reloaded artifact, preserving the same two
intermediate checkpoints. The full half-precision tiny-Qwen test passes, including
reference-file hashing and base restoration; all 15 focused tests remain green.

[Protocol v1](../experiments/instruction_boundary/distribution_retention_protocol_v1.json)
fixes one fit, weight 1, 300 generated records including replay, a 64 MiB reference
cache cap, and a 5,400-second provider limit. No GPU submission has been made at
this checkpoint. The protocol's memory estimate is from tokenization/config,
not a measured large-model allocation. Independent artifact verification and
blinded answer review remain required after execution.

## Accepted run

One private T4 run is active: [Kaggle version 1](https://www.kaggle.com/code/shlydv/neuronpatch-distribution-retention-v1-20261025), kernel 140750471. Accepted source matches the local bundle byte-for-byte, SHA256 `d5a5c0130ea993cf82b7880ef076186290e8793a876b71a9f35bb885b5995449`. Source revision: `b3477f5`. Provider limit: 5,400 seconds. Fifteen focused tests passed, including the full tiny-model lifecycle and rejection of a corrupted cache. No large-model result is available yet.


## Related work checked during execution

[Refining Salience-Aware Sparse Fine-Tuning (ACL 2025)](https://aclanthology.org/2025.acl-long.1541/)
reports competitive simple gradient-based static masks and no substantial dynamic-mask
benefit in its experiments. This supplies a simpler baseline to consider, not evidence
that activation probes or added selector complexity will improve our task.

[NGFT (September 2026), section 3.3](https://arxiv.org/html/2609.05913v1)
uses an MSE anchor derived from mean FFN activations on its instruction dataset,
introduced after CE converges. Its appendix compares representation drift and its
evaluation includes other tasks. This is a different preservation objective from
our per-context output KL. A mean activation anchor need not equal an individual
example's original activations; retaining the mean does not mathematically imply
retaining each answer. This last statement is our interpretation, not the paper's
claim. The paper and two pinned core source files were inspected; no full upstream training reproduction was run. See the source observations below.

Decision: retain the running one-factor control. Do not add activation anchoring,
dynamic masks or data selection to it. If useful learning survives with fewer
regressions, verify fresh transfer before a matched independent-row random/gradient/
guided comparison. If KL prevents both damage and learning, report that tradeoff.
If damage persists despite small training KL, examine context coverage rather than
claiming preservation has been established by a low scalar loss. None of these
outcomes alone establishes an advantage over existing sparse-tuning libraries.


### NGFT released-source observations

Inspected `NLP2CT/NGFT` commit `77537f90adba9d4cd33a84cfbc8add10cc4c926e`.
The [trainer's mask block](https://github.com/NLP2CT/NGFT/blob/77537f90adba9d4cd33a84cfbc8add10cc4c926e/trainer.py#L2573)
uses a space in the up-projection filter. Evaluating the exact parsed predicate
on a standard `up_proj` name skips that block. Its substring layer predicate
matches indices 0, 1 and 10 for layer 10, applying multiple masks successively.
Other projection gradients skip this mask block too; whether they update depends
on their trainability/optimizer setup elsewhere, which this audit does not establish.

The alignment path at lines 3817–3873 uses an epoch-step condition, a hard-coded
BioInstruct anchor filename and an imported similarity function, rather than
transparently exposing the paper's CE-threshold/MSE recipe. Its hook path averages
across sequence positions and builds an all-ones attention mask for generated
inputs. These are implementation differences to resolve before a faithful port,
not evidence that the published experiments used this exact path or that their
results are invalid. Two AST predicates were executed on synthetic names; no
upstream trainer, external pickle or model was executed. Pinned source hashes and
the predicate receipt are in private `artifacts/ngft_source_audit`.

Decision: do not adopt this trainer wholesale or treat its README as executable
specification. Any future neuron-anchoring control must specify axis identities,
masking, exact frozen references, loss direction and activation timing explicitly.
The currently running output-KL control is unaffected.
