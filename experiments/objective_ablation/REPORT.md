# Explicit preservation changes the trade-off

Exploratory analysis, 2026-10-19. These are the previously exposed preservation_qa
examples and frozen supports, not fresh confirmation. All fits use lr 0.0003,
32 steps and identical training order. This is not a reproduction of FiNE.

| Parameterization | KL coefficient | Supported correct / 48 | Missing correctly abstained / 48 |
|---|---:|---:|---:|
| Activation-selected columns | 0 | 41 | 46 |
| Activation-selected columns | 0.1 | 42 | 46 |
| Activation-selected columns | 1 | 44 | 42 |
| Random columns, seed 11 | 0 | 47 | 42 |
| Random columns, seed 11 | 0.1 | 47 | 42 |
| Random columns, seed 11 | 1 | 47 | 42 |
| LoRA | 0 | 43 | 43 |
| LoRA | 0.1 | 44 | 43 |
| LoRA | 1 | 47 | 43 |

All runs retain 16/16 arithmetic accuracy. Column-patch reloads have zero fixed-input
logit error. These small differences are exploratory observations from one training
seed; no confidence or general superiority claim is justified.

## Interpretation

Changing only the objective changes over-abstention. For selected columns, weak KL
recovers one supported answer without losing correct abstentions; stronger KL
recovers three but loses four correct abstentions. LoRA improves supported accuracy
at both strengths without changing abstention accuracy. Random-column discrete
accuracy is unchanged at this resolution; that does not imply unchanged logits.

The objective deserves consideration independent of neuron localization. A discovery
method must earn its advantage when both it and its baselines receive preservation
training. Our old comparisons cannot establish that a paper-inspired selector plus
its intended objective would fail. Equally, simply adding KL has not established a
superior neuron method: LoRA with KL remains a strong comparator.

Do not promote coefficient 1 as a default based on these exposed examples. Follow-up
must explore coefficient, support and intervention-axis effects on development
folds, then verify an improvement on independent data. This evidence supports
adding explicit preservation to our design space; it does not resolve which
combination is best or prove that the objective causes broad robustness.

## Implementation and provenance

`neuronpatch.losses.preservation_kl` computes either direction explicitly, detaches
the teacher, masks prediction positions, and averages equally across examples.
This experiment uses KL(reference || edited) on response-prediction tokens of
supported examples only. Reference forwards use the exact original checkpoint in
eval mode. Ordinary supervised response loss remains unchanged.

The executed source is `results/executed_run.py.txt`; its SHA-256 matches summary's
`code_sha256`. After launch, the runner's reference cleanup was changed from
`del reference` to `reference = None` to avoid a closure lint warning; the snapshot
retains the exact code used. This does not change any training or evaluation step.
Model revision, versions, raw outputs, per-step CE/KL, timing and support indices
are in `results/summary.json` and adjacent files. Coefficient-zero KL entries are
zero because that term was not computed, not because measured divergence was zero.

Run from repo root with `HF_HOME=/path/to/cache python experiments/objective_ablation/run.py`
after installing `.[dev,experiment]`. Requires Apple MPS and the previous experiment's
published data/summary. Move existing results aside; the runner refuses overwrite.
