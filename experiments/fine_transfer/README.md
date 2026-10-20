# Cross-prompt gradient selection: bounded development pilot

**Status: submitted to Kaggle, version 1, eight new fits; results pending.**
[Private run](https://www.kaggle.com/code/shlydv/neuronpatch-fine-gradient-transfer-v1-20261020),
kernel ID 139844236. This is not a method improvement claim.

The verified [FiNE reference](../fine_reference/reference16_v1/REPORT.md) changes
15/16 original answer prefixes but only 5/16 relation rewordings. Its selector
scores contributions to the base model's predicted token. We test whether update
gradients that agree across phrasings offer a better trainable support than
single-prompt attribution. Extra training data is a separate control.

## Mechanism

For one prompt c and a scalar target-token CE, a linear projection's column j
has gradient g_cj = B_c^T A_c[:,j], with A its inputs and B the loss derivative
of its outputs. This includes all sequence positions and the downstream model,
rather than only a final-token activation/unembedding approximation. Define
the mean gradient g_bar_j over three supervised phrasings. The score is

    min_c <g_cj, g_bar_j> / ||g_bar_j||

with zero assigned to zero-norm mean gradients. It estimates the worst prompt's
first-order improvement along a unit step opposite the mean gradient. It does
not guarantee finite-step behavior, Adam behavior, preservation or transfer.

`gradient.py` computes these inner products using small token-position Gram
matrices, avoiding dense full-model parameter gradients. Frozen embedding
outputs are differentiable; a constant loss scale of 128 limits FP16 backward
underflow, and derivatives are unscaled before scoring. Float64 CPU reductions
are used. Tests compare against independently materialized autograd gradients,
including a real tiny GPT-J model. Opposed and zero mean gradients are explicit.

Gradient attribution already appears in [KNE](https://arxiv.org/html/2412.20637v1).
[NMKE](https://arxiv.org/html/2510.22139v1) uses cross-prompt attribution patterns;
[NeST](https://arxiv.org/html/2602.16835v2) motivates shared updates using gradient
alignment within clusters. This experiment does not reproduce those methods or
claim novelty for gradient selection or cross-prompt consistency. Its purpose is
to test a concrete update-suitability signal against our verified baseline.

## Frozen comparison

[Protocol](protocol_v1.json), [exact inputs](inputs_v1.json),
[submitted source/data hashes](submission_v1.json).

- First two exposed KnowEdit records; no new outcome-based case selection.
  Original FiNE artifacts already have exact full-size library/source parity.
- Each case has the original prompt plus two newly authored training phrasings.
  The official rephrase and two further new wordings are not trained on.
  New wordings are assistant-authored development material, not independent gold
  or a fresh confirmation set. All six held-out-wording probes are reported.
- A saved original FiNE patch is the single-prompt reference. Its full known
  generations must reproduce exactly before new fits for the case.
- Four new fits per case: FiNE supports, single-prompt actual-gradient supports,
  three-prompt gradient-agreement supports, and consensus-layer-matched random.
  All four use the same three training prompts and target, objective and rate.
  The random control also preserves the selected support's duplicate pattern.
- Five occurrences per target token, 4096 BF16 parameters per occurrence;
  identical scalar parameter budgets across methods within a case. Repeated
  occurrences remain independent parameters, as in FiNE.
- Common objective: worst token/prompt NLL, mean prefix KL and repetition penalty,
  one global norm penalty. Adam lr .001, at most 50 updates, original coefficients
  and stopping threshold. Single-prompt numerical equivalence to our verified
  FiNE workflow is tested; multi-prompt training is an explicit adaptation.
- Same pinned GPT-J-6B weights, FP16 base, Torch 2.4.1, Transformers 4.40.0,
  greedy 40-token generations, original sequential merge order. Restore checks
  after every condition and full base fingerprint at start/end.
- Thirty-two fixed untrained utility probes: capitals, arithmetic and elementary
  facts. Report all baseline outputs and retain only baseline-reference-correct
  probes in conditional preservation denominators; never call unchanged wrong
  answers preserved utility. The source's relation probes are also retained.
- Discovery has extra forward/backward/CPU scoring cost, reported separately;
  equal trainable parameters do not mean equal total compute. No learning-rate
  sweep. Ninety-minute worker budget checked between fits, with an outer 95-minute
  worker timeout; package setup is separate.

The frozen advancement gate requires both original edits to work, at least two
additional held-out-wording prefix hits versus **both** augmented FiNE and
single-gradient selection, and no additional lost baseline-correct utility
answers versus either. At least eight utility probes must work at baseline.
Loss sets are compared per case/probe, not only as aggregate counts. A pass is
permission to investigate fresh cases and stronger low-rank controls, not a
claim to have surpassed a paper. Matched-random results and complete continuations
must also support interpreting an identity-selection benefit. If augmented FiNE
does as well, the gain cannot be credited to the selector.

The analysis guards numeric utility prefixes: `4.5`, `4,000` and `4e2` must not
count as `4`. This guard was added before inspecting any pilot answer outputs;
the frozen data and requested answers were not changed. Per-fit times include
export and generation and are not pure training-speed measurements. Discovery
forward/backward counts and available console timestamps must be reported
alongside those times; no efficiency gain is assumed.

No utility loss is trained in this screen beyond FiNE's prefix KL. It cannot
show our complete correction/preservation recipe works. Prefix matching can
accept a correct start followed by nonsense or miss a valid elaboration; inspect
full outputs, especially repetition and relation spillover. LoRA and fresh-case
comparisons remain required before broader claims. Do not tune on pilot outcomes.

## Rebuild and analyze

`prepare.py` accepts the pinned original dataset and our four-artifact parity
directory, verifies hashes and writes only the two selected artifacts and frozen
inputs. The public protocol records their hashes; no full model weights or
credentials enter `build.py`'s private Kaggle bundle. Model and environment caches
live under `/tmp`, outside saved Kaggle outputs.

```sh
python -m experiments.fine_transfer.prepare \
  --source /path/to/pinned-test_cf.json \
  --artifacts /path/to/parity-results --output /tmp/transfer-data
python -m experiments.fine_transfer.build \
  --data /tmp/transfer-data --output /tmp/transfer-bundle
python -m experiments.fine_transfer.analyze \
  --results /path/to/downloaded/results --output /tmp/transfer-analysis
```

Before interpreting downloaded results, run the independent comparison verifier:

```sh
python -m experiments.fine_transfer.verify \
  --download /path/to/transfer-results --decode --output /tmp/transfer-integrity.json
```

Use the pinned Transformers 4.40 tokenizer environment with `--decode` to match
the worker's decoding defaults. This checks submitted source/data identity,
reference generations, paired inputs/base outputs, all eight fits, scalar budgets,
random-control duplicate patterns, artifact tensors and every decoded sequence.
It reports unique columns and summed update norms as well as occurrence counts:
equal advertised trainable scalars need not mean equal effective support.

## Prospective subject-scope diagnostic

While v1 was running, before reading any pilot answer outputs, we prepared
[16 other-subject prompts](scope_inputs_v1.json) with identical relation templates.
These are the first eight matches per edited case at official source indices
128 onward, excluding the edited subject and references matching the new target.
Indices16–127 remain reserved. The raw input bytes were fixed on first preparation;
the execution rule was clarified before any answer review. This is a separate
prospective diagnostic on two exposed edits, not independent method confirmation.

The [scope protocol](scope_protocol_v1.json) tests whether the patch makes other
people share the edited citizenship, or other entities share the award's edited
country association. Perfect capitals/arithmetic utility cannot rule out such
relation-wide leakage. References are benchmark labels rather than independently
validated current facts. Report baseline-correct retention and target leakage
separately, including cases where the base was already wrong.

No scope GPU run has been submitted. Review the complete first pilot and its
declared execution condition first; do not overlap workers or retune the saved
patches. Every v1 condition must be evaluated if the diagnostic goes ahead.

The experimental modules are deliberately separate from the public adaptation
API until model behavior justifies a recommended workflow.
