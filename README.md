# neuronpatch

Train compact updates to selected neuron output weights, evaluate what changes,
and export a patch tied to its exact base checkpoint.

**Status: experimental foundation.** Version 0.1 implements a tested PyTorch
weight-update primitive, not a validated behavioral editing system. No LLM
benchmark results, automatic causal discovery, or superiority to LoRA are claimed.

## Direction

The intended workflow is **select → train → evaluate target and preserved behavior
→ export**. The first application will be document-grounded answering: answer
supported questions and abstain when evidence is missing, without becoming less
helpful on answerable questions. The user supplies both correction examples and
examples of behavior to preserve.

We build on published ideas rather than claiming a new fine-tuning algorithm.
See [research provenance](docs/research.md) for what is implemented, what is only
planned, source links, and reproduction status.

## What works today

- Explicit selection of input columns of an ordinary `torch.nn.Linear`.
- Compact trainable deltas; every original model parameter is frozen.
- Contrastive activation ranking with explicit token/padding masks and equal example weights.
- Deterministic, without-replacement random selection for per-layer controls.
- Safetensors patch export with exact base-weight fingerprint validation.
- A merged copy with ordinary linear layers and no inference hooks.
- CPU tests for gradients, frozen weights, column locality, serialization,
  merging, invalid inputs, and learning a known synthetic edit on held-out data.

## Install and run

Python 3.10+; no model download, API key, or GPU is needed for the toy example.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
python examples/fit_toy.py
```

## Core API

```python
import torch

from neuronpatch import NeuronPatch

# model is an already-loaded, floating-point PyTorch model.
# For a Qwen-style MLP, down_proj input columns correspond to intermediate neurons.
patch = NeuronPatch(
    model,
    {
        "model.layers.12.mlp.down_proj": [14, 98, 203],
    },
)
optimizer = torch.optim.AdamW(patch.trainable_parameters(), lr=1e-3)

# Run your normal forward/loss/backward/optimizer loop here.
# The indices and learning rate above are illustrative, not a recommended recipe.
patch.save("artifacts/my-patch", metadata={"base_revision": "exact model revision"})
merged = patch.merged_model()

# fresh_base must have exactly the original weights, buffers, and dtype.
restored = NeuronPatch.load(fresh_base, "artifacts/my-patch")
```

`NeuronPatch` modifies the supplied model in place and freezes its original
parameters. `merged_model()` returns a separate model; it needs memory for a copy.
Use the untouched original checkpoint to roll back. Repeated merge/subtract cycles
are deliberately avoided because of floating-point rounding.

## The math

PyTorch stores a linear weight as `W[out_features, in_features]`. For selected
input features `I`, we learn `D[out_features, len(I)]`:

```text
y = x Wᵀ + b + x[..., I] Dᵀ
W_merged[:, I] = W[:, I] + D
```

In an MLP down-projection, those features are the intermediate neuron activations.
Selecting 32 neurons with output width 4096 trains 131,072 scalar weights—not 32.
Zero initialization leaves the base function unchanged. The merged and unmerged
forms are mathematically equivalent; floating-point output differences are expected.

## Boundaries

Only ordinary, materialized floating-point `nn.Linear` targets are supported.
Quantized/fused projections, tied/shared target parameters, distributed training,
optimizer resume, and patch composition are not supported yet. The generic primitive
and a randomly initialized tiny Qwen3 architecture are tested on CPU. Pretrained
Qwen behavior and other model families have not been validated.

Keep base weights and persistent buffers unchanged after attachment. Models with
training-time mutable buffers (e.g. BatchNorm) require special handling and are not
supported by this checkpoint contract. The fingerprint checks tensor bytes, not
architecture code or tokenizer compatibility; record and pin those separately.
Computing it transfers one tensor at a time to CPU and adds startup cost.

Small patches do not imply proportional training-speed improvements. We do not
claim that disjoint edits are behaviorally independent or that selected neurons
are causal merely because their activations correlate with a label.

## Next milestones

1. **Foundation (implemented):** column deltas, export, CPU correctness tests.
2. **Architecture integration (implemented):** tiny random Qwen3 with supervised
   token loss, fresh-input export parity, and exact patch reload. A pinned small
   pretrained checkpoint is next.
3. **Guided selection (initial heuristic implemented):** collect actual
   down-projection inputs with explicit token masks, average within examples and
   then across examples, and rank the absolute difference between group means.
   This is not FiNE attribution or causal discovery. Behavioral comparison with
   layer-matched random controls remains to be run.
4. **First behavior recipe:** document-grounded QA, correction/preservation data,
   validation-only tuning, held-out document families, abstention and utility metrics.
5. **Baselines and publication:** prompting, matched random-column patches, ordinary
   LoRA; report seeds, compute, data budgets, patch sizes, and negative results.
6. **Additional methods:** evaluate NeST-inspired cluster-shared updates and
   SafeNexus-inspired masked LoRA as separately named implementations.

Full paper reproductions must pin models, datasets, objectives, and evaluations.
An inspired primitive is not a reproduction. First benchmark acceptance criteria
will be recorded before running it; this release establishes numerical correctness.

## Development

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
```

Small, reviewable changes are preferred. Include tests for behavioral invariants,
not only tensor shapes. AI-assisted contributions should disclose assistance and
be reviewed by the contributor. Do not commit model weights, credentials, or datasets.

MIT licensed. Research attribution is in [docs/research.md](docs/research.md).

## Offline Qwen3 integration and guided selection

```bash
python -m pip install -e '.[dev,transformers]'
HF_HUB_OFFLINE=1 python examples/fit_tiny_qwen.py
```

This initializes a tiny Qwen3 from configuration, trains 256 delta parameters on
synthetic token IDs, and verifies export/reload on fresh inputs. Training-loss
reduction is a plumbing check, **not evidence of language understanding, guided
selection superiority, or behavioral generalization**. No weights are downloaded.

The selector API separates calibration from optimization:

```python
from neuronpatch import mean_activations, contrastive_indices, random_indices

# Each iterable yields (model_input_dict, boolean_selection_mask).
# Masks are [batch, sequence]; select the same token roles in both groups.
positive = mean_activations(model, targets, positive_batches)
negative = mean_activations(model, targets, negative_batches)
selected = {name: contrastive_indices(positive[name], negative[name], count=32)
            for name in targets}
control = {name: random_indices(positive[name].numel(), 32, seed=42)
           for name in targets}
```

`mean_activations` uses streaming statistics rather than saving complete activation
caches. It temporarily enters eval/inference mode, restores module training flags,
and removes hooks even after errors. Padding must be excluded; every example must
select at least one token. Use full teacher-forced forwards, not cached generation.
Keep calibration data separate from the final behavioral test. Tokenization,
response masks, batch construction, and input device placement are caller-owned.
