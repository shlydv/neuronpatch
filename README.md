# neuronpatch

Train compact updates to selected neuron weights, evaluate what changes,
and export a patch tied to its exact base checkpoint.

**Status: experimental foundation.** Version 0.1 provides tested PyTorch primitives
for compact weight updates, supervision, export and reload. We have **not**
demonstrated a better adaptation method or a recommended behavioral editing recipe.

| Evidence | What is verified | What it does not establish |
|---|---|---|
| [Upstream FiNE reference](experiments/fine_reference/reference16_v1/REPORT.md) | Requested prefixes on 15/16 original prompts versus 0/16 for random; 5/16 rewordings | Robust complete-answer editing or a new method |
| [Matched-budget Qwen3-4B comparison](experiments/paired_mlp/complete_v1/REPORT.md) | Selected guided, random and LoRA conditions learn all 8 transfer answers across two facts; guided updates also cause unrelated errors | A guided-selection advantage |
| [Stopping and deployment checks](experiments/paired_checkpoints/complete_v1/REPORT.md) | Earlier stopping preserves the edits, but the guided errors persist; the separate replay rules out merging on these endpoints | A safe or cheaper guided recipe |
| [Functional-direction calibration](experiments/functional_selection/complete_v2/REPORT.md) | Risk-aware selection lowers held-out drift for similar one-step learning gain; finite risk prediction fails | A reliable risk certificate, completed edits, or a trained advantage over LoRA |
| [Measured-preservation training](experiments/guarded_training/complete_v1/REPORT.md) | All controls stall or fail transfer; the guard also penalizes valid answer aliases | A working protection rule or selected-neuron advantage |
| [Outcome-guard replay](experiments/outcome_guard/complete_v1/REPORT.md) | Correct greedy controls hide increased wrong-continuation probability; canonical auxiliary labels impose avoidable EOS pressure | A qualified protection rule, causal explanation of downstream errors, or improved trained method |
| [Auxiliary-objective ablation](experiments/alias_supervision/complete_v1/REPORT.md) | Accepted-set supervision helps random/LoRA oxygen probabilities and selected Lakers answers; selected oxygen error persists | A selected-neuron advantage or a generally safe adaptation recipe |
| [Instruction-boundary comparison](experiments/instruction_boundary/complete_v2/REPORT.md) | Compact patches train and deploy; one random support recovers 19/27 unseen-authority failures versus 21/27 for LoRA and 0/27 for guided selection | A reliable selection recipe, broad security guarantee, or training-speed advantage |
| [Fixed-support supervision repair](experiments/instruction_boundary/complete_supervision_v1/REPORT.md) | Full-document legitimate compliance restored, but unseen attack resistance weakened for guided, random and LoRA | A combined repair, selected-neuron advantage, or robust boundary learner |

The [completed relation-learning pilot](experiments/relation_learning/complete_v2/REPORT.md)
raises held-out task accuracy from 39/64 to 64/64 with LoRA and random patches,
and 63/64 with Fisher-selected patches. Selection has no demonstrated advantage;
all rates and preservation failures are reported.

The [eight-task-example follow-up](experiments/relation_learning/complete_low_data_v1/REPORT.md)
also finds no Fisher advantage: selected LoRA reaches 126/128, Fisher 121/128,
and random supports 112/128 and 122/128. Transfer wording reveals errors that
original-wording validation loss misses; all rates and control failures are retained.

Start with the [current method decision](docs/current-method.md) for the active
hypothesis and decision boundary. The [experiment history](docs/experiment-history.md)
preserves the earlier transfer, preservation and failure investigations, with
links to their protocols and full reports. Numerical correctness and a successful
reproduction are evidence for their stated scope, not proof of an improved recipe.

## Direction

See the [current method decision](docs/current-method.md) for the evidence,
paper-informed choices, completed endpoint checks and next experiment boundary.

The intended workflow is **select → train → evaluate target and preserved behavior
→ export**. The current research application is document-grounded answering:
use document facts while respecting the user's instructions, preserve correct
answers, and avoid unsupported answers. Earlier abstention experiments and the
current instruction-boundary work have not established a recommended recipe.
The user supplies both correction examples and examples of behavior to preserve.

We build on published ideas rather than claiming a new fine-tuning algorithm.
See [research provenance](docs/research.md) for what is implemented, what is only
planned, source links, and reproduction status.

## What works today

- Explicit selection of input columns or output rows of an ordinary `torch.nn.Linear`.
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

The core recipe is checked on PyTorch 2.4.1 without Transformers or PEFT;
model-dependent tests skip when those optional packages are absent. A separate
CI job covers this supported older runtime. See [compatibility checks](docs/compatibility.md).

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
and a randomly initialized tiny Qwen3 architecture are tested on CPU. A small pretrained
Qwen2.5 pilot is reported below; broad behavior and other families remain unvalidated.

Keep base weights and persistent buffers unchanged after attachment. Models with
training-time mutable buffers (e.g. BatchNorm) require special handling and are not
supported by this checkpoint contract. The fingerprint checks tensor bytes, not
architecture code or tokenizer compatibility; record and pin those separately.
Computing it transfers one tensor at a time to CPU and adds startup cost.

Small patches do not imply proportional training-speed improvements. We do not
claim that disjoint edits are behaviorally independent or that selected neurons
are causal merely because their activations correlate with a label.

## Next milestones

1. **Foundation (implemented):** row/column deltas, export, merge and correctness tests.
2. **Architecture integration (implemented):** tiny Qwen correctness tests and
   experiments with pinned pretrained Qwen2.5-0.5B-Instruct and Qwen3-4B-Instruct-2507.
   Behavioral robustness
   across model families remains unproven.
3. **Guided selection (initial heuristic implemented):** collect actual
   down-projection inputs with explicit token masks, average within examples and
   then across examples, and rank the absolute difference between group means.
   This is not FiNE attribution or causal discovery. Synthetic and real-passage
   comparisons have not established an advantage over random selection.
4. **First behavior recipe (experimental):** document-grounded QA with isolated
   article splits, correction/preservation data, answer/refusal metrics and output
   audits. Preservation failures still prevent recommending a default recipe.
5. **Baselines and publication:** prompting, matched random-column patches, ordinary
   LoRA; report seeds, compute, data budgets, patch sizes, and negative results.
6. **Additional methods:** evaluate the tested experimental shared-row primitive and
   SafeNexus-inspired masked LoRA as separately named implementations.

Full paper reproductions must pin models, datasets, objectives, and evaluations.
An inspired primitive is not a reproduction. Experiments distinguish exploration
from fresh confirmation; numerical correctness does not establish useful adaptation.

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
selected = {name: contrastive_indices(positive[name], negative[name], count=32) for name in targets}
control = {name: random_indices(positive[name].numel(), 32, seed=42) for name in targets}
```

`mean_activations` uses streaming statistics rather than saving complete activation
caches. It temporarily enters eval/inference mode, restores module training flags,
and removes hooks even after errors. Padding must be excluded; every example must
select at least one token. Use full teacher-forced forwards, not cached generation.
Keep calibration data separate from the final behavioral test. Tokenization,
response masks, batch construction, and input device placement are caller-owned.

## First pretrained pilot

A bounded Qwen2.5-0.5B-Instruct experiment is complete:
[protocol and results](experiments/grounded_qa/REPORT.md).
The selected patch improved correct abstention from 13/24 to 24/24 but reduced
supported answers from 24/24 to 17/24. It **failed the preservation criterion** and
performed worse than the random-column controls and same-layer LoRA. This validates
pretrained training/export plumbing, not the selection heuristic's usefulness.
Raw predictions and configurations are published; no superiority claim is made.

A [second, fresh-data preservation experiment](experiments/preservation_qa/REPORT.md)
compared gradient-based selection, activation contrast, three random seeds and
LoRA with equal learning-rate search budgets. Both guided selectors failed the
preservation gate again. All three random controls passed this small synthetic
pilot, but that is not evidence of real-document robustness. The new selector and
task-specific no-edit validation policy remain experimental, outside the core API.

The ongoing [method-design audit](docs/method-design.md) separates the papers'
selection, intervention axes, capacity constraints and training objectives. An
[exploratory objective ablation](experiments/objective_ablation/REPORT.md) shows
that explicit distribution preservation can improve the trade-off, including for
LoRA. This is not fresh confirmation or evidence of superiority. The tested loss
primitive is available as `neuronpatch.losses.preservation_kl`; callers explicitly
supply aligned prediction-position masks and choose the KL direction.

## Selecting input columns or output rows

`NeuronPatch` defaults to `axis="input"`, preserving the original down-projection
column API. Use `axis="output"` for selected output rows, such as Qwen gate/up
projections:

```python
patch = NeuronPatch(
    model,
    {"model.layers.12.mlp.gate_proj": [14, 98, 203]},
    axis="output",
)
```

Rows change the selected linear output features; columns change the contribution
of selected inputs. In a gated MLP these are different interventions. Row patches
freeze biases and train `len(indices) * in_features` weights. Selection and learning
rates remain caller decisions. A patch uses one axis across its targets; mixed-axis
patches are not supported yet. Row artifacts use format 2; legacy column artifacts
remain format 1 and are still loadable. Merge/export/reload are supported for both.

An [intervention-axis ablation](experiments/axis_ablation/REPORT.md) found a promising
up-row candidate that preserved all supported answers on the exposed synthetic
set. Its advantage over matching random up rows was only one answer. Treat it as
exploratory evidence, not a validated default or a claim of superiority.

## Current real-passage evidence

The [completed data-size curve](experiments/real_qa/scale_2048/REPORT.md) improves
some answer/refusal trade-offs, but still loses supported answers. Expanding
[preservation coverage](experiments/real_qa/coverage_512/REPORT.md) does not solve
that problem. A [decoding audit](experiments/real_qa/decoding_512/REPORT.md) identifies
an inherited repetition penalty that contributes to both useful and unnecessary
refusals; generation settings must be explicit in comparisons.

`preservation_kl(..., token_reduction="max")` is an experimental alternative to
the default per-example token mean. It protects the largest token-distribution
change per example; examples still receive equal weight. It is being compared
with stronger mean KL, not presented as an effective default. The motivation and
controls are in the [token-loss protocol](experiments/real_qa/coverage_512/TOKEN_LOSS_PROTOCOL.md).

`neuronpatch.grouped.GroupedRowPatch` provides explicit shared row deltas with
verified forward/gradient/merge behavior. Use `NeuronPatch(..., axis="output",
groups={module_name: group_labels})` to attach and save grouped rows through the
high-level API. Each target needs one group label per selected index; labels are
contiguous from zero. For example, indices `[5, 1, 3]` and groups `[1, 0, 1]` share
one update between rows 5 and 3 and give row 1 another. Artifacts use format 3;
legacy row/column artifacts remain supported. There is no automatic grouping.
For a half-precision base, explicit `delta_dtype=torch.float32` keeps grouped
trainables and their update computation in FP32; the update is cast at the output
boundary. These artifacts use format4 and preserve the adapter dtype on reload.
Merging adds in promoted precision before casting once to the base weight dtype.
Live and merged outputs can still differ through rounding, so test deployment
behavior separately. Omitting this option retains the original dtype behavior.
Neither this primitive nor
the new loss has demonstrated an advantage in model quality. See the
[evidence ledger](docs/method-design.md) for the remaining work.

### Per-token contribution selection (experimental)

`contribution_indices` implements the all-eligible-layer contribution score used
by FiNE's knowledge editor. Given captured activations `[positions, layers, FFN]`,
one down-projection weight `[hidden, FFN]` per layer, an unembedding matrix
`[vocabulary, hidden]`, and the base model's predicted token at each position, it
returns `[positions, k, 2]` indices `(layer, neuron)`.

```python
from neuronpatch import contribution_indices

supports = contribution_indices(
    activations,
    down_weights,
    unembedding,
    predicted_tokens,
    k=5,
    exclude_last_layers=3,
)
```

The default FP16 scoring matches the upstream scoring precision; precision and
backend can affect ties/rankings. Tensor capture and causal position alignment
are caller responsibilities. This function is a selector, not a complete FiNE
editor. Its indices exactly match the unchanged upstream scoring routine on
18 CPU fixtures (72 token positions). The
[full-size workflow check](experiments/fine_reference/library_parity_v1/REPORT.md)
also matches upstream on two exposed 6B-model cases with both selection methods.
See `experiments/fine_reference/check_contribution.py` and its checked result.

Preserve repeated selections across target positions when reproducing the editor.
Its separate deltas and norm penalty can differ from training one deduplicated
parameter per neuron. Existing `NeuronPatch` column indices remain unique; summing
learned deltas for export and deduplicating supports before training are distinct
operations. Shared updates across related neurons are a separate hypothesis.

### Occurrence-preserving column edits (experimental)

`OccurrenceColumnEdit` preserves FiNE's separate update vector for every selected
occurrence, including repeated selections of the same column. It owns one global
`delta` matrix and exposes its joint norm; summing per-layer norms would change
the reference objective. Target projections must have the same output width and
share a device. An explicit delta dtype supports the reference's FP16 base and
BF16 updates.

```python
import torch
from neuronpatch.occurrence import OccurrenceColumnEdit

edit = OccurrenceColumnEdit(
    model,
    [("transformer.h.19.mlp.fc_out", 5051), ("transformer.h.19.mlp.fc_out", 5051)],
    delta_dtype=torch.bfloat16,
).eval()
optimizer = torch.optim.Adam(edit.trainable_parameters(), lr=0.001)
# Compute the chosen task loss from edit(...); delta_norm() is the joint penalty.
# For generation, keep hooks active around the complete generation call:
with edit.applied() as edited_model:
    output = edited_model.generate(**inputs)
```

Contexts remove hooks on exit, including exceptions. Do not use concurrent edits
on the same model. `merged_model()` deep-copies the base and adds occurrences in
order, requiring memory for a second model. Live mixed-precision hooks and merged
weights can have different rounding. Use `edit.save(path)` and `OccurrenceColumnEdit.load(original_model, path)`
for compact artifacts. They retain ordered duplicate occurrences and delta dtype,
and verify the original base weights/dtype. They exclude optimizer state. The
ordinary state dictionary still includes the frozen base model.

On a tiny CPU GPT-J with 256 occurrences / 128 unique features, four Adam steps
produce exactly the same BF16 deltas as the unchanged upstream FiNE trainer,
for both FP32 and FP16 base weights. Merged weights also match the reference
sequential updates exactly and original weights remain unchanged. See
`experiments/fine_reference/check_edit.py` and the two `edit_checks_*.json` files.
These are numerical integration checks, not a new behavioral result.
The separate 6B reference benchmark still uses upstream.

### Reference knowledge-editing loss (experimental)

`neuronpatch.knowledge.knowledge_edit_loss` implements the published FiNE-K code's
single-sequence objective: maximum target-token NLL, reference-to-edited prefix
KL, a post-target repetition penalty, and the joint occurrence-delta norm. It
returns the total and individual weighted components. Caller-provided logits
must include the complete prompt and target with their causal alignment; the
reference prefix excludes the prediction position for the first target token.
This helper deliberately does not claim to implement FiNE-S safety editing or a
batched fine-tuning loss. Its KL direction follows the implementation, which
differs from the paper's written direction.

```python
from neuronpatch.knowledge import knowledge_edit_loss

loss = knowledge_edit_loss(
    edited_logits,
    reference_prefix_logits,
    target_tokens,
    edit.delta,
    alpha=1.0,
    beta=10.0,
    gamma=0.1,
)
loss.total.backward()
optimizer.step()
```

The integrated CPU source check now uses the library selector, objective and
occurrence update together. All selected indices and learned deltas match the
unchanged upstream trainer exactly for the tested FP32/FP16-base configurations.
Scope remains four steps on a tiny random model with repeated selections.
The workflow below also verifies token/activation capture and stopping. Full-size
GPU parity now passes on two exposed cases and both selection methods; the
16-case quality benchmark ran the original editor independently.

Compact occurrence artifacts use separate `edit.json` / `edit.safetensors` files;
they do not change the legacy `NeuronPatch` formats. Loading validates target
shapes, tensor dtype/values and the exact base fingerprint before freezing the
model. Saving rejects a changed base, active edit context, or nonfinite deltas,
and requires a fresh output directory. The integrated upstream check includes
exact delta/logit reload and sequential-merge parity for both tested precisions.

### Complete GPT-J reference workflow (experimental)

`fit_gptj_edit` captures activations, selects per-token occurrences, trains the
reference objective, and returns a compact edit. It uses an ordinary Hugging Face
GPT-J model and removes its scoped hooks after every forward. Load the model and
tokenizer yourself; they must use compatible pinned revisions. This single-device
workflow requires a materialized floating-point model and enough memory for
backpropagation. It does not load or download models implicitly.

```python
import torch
from neuronpatch.editing import GPTJEditConfig, fit_gptj_edit

result = fit_gptj_edit(
    model,
    tokenizer,
    prompt,
    desired_answer,
    GPTJEditConfig(neurons_per_token=5, max_steps=50),
)
result.edit.save("artifacts/gptj-edit", metadata={"model_revision": revision})
# Reuse the exact training prompt context, excluding the desired answer.
prompt_ids = result.input_token_ids[: -len(result.target_token_ids)]
input_ids = torch.tensor([prompt_ids], device=next(model.parameters()).device)
with result.edit.applied() as edited_model:
    output = edited_model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
```

The routine prepends GPT-J's EOS/BOS and one target space, matching upstream;
it performs no truncation or chat templating. Success freezes the original model
in eval mode; failure restores original training and gradient flags. The edit
does not mutate original weights. Random controls use `selection="random"` and
retain the reference's sampling with replacement; this differs from the generic
without-replacement selector. History records losses before each possible update,
so early stopping can produce zero optimizer steps.
Keep evaluation conditioning explicit: the upstream GPT-J metric omits the
prepended EOS used by the editor and our generation diagnostic. The two paths
are retained for reference comparison; a product evaluation should compare
generated behavior and token scores under the same prompt context.

The ordinary-HF workflow exactly matches upstream supports, learned BF16 deltas
and sequential merged weights on tiny CPU models with FP32/FP16 bases, both
selection methods, explicit EOS, and zero-step stopping. Checks and runtime details
are in `experiments/fine_reference/edit_checks_*.json`. The
[6B GPU comparison](experiments/fine_reference/library_parity_v1/REPORT.md) also
matches ordered supports, learned deltas, merged columns and fixed merged
generations exactly in all four conditions. It uses the explicitly isolated
torch 2.4.1 / Transformers 4.40.0 reference runtime; the optional general
integration extra has a different minimum version. This is a faithful baseline,
not an improved method or broad model/version validation.

Live mixed-precision hooks can produce different continuations from merged
weights: one of 18 fixed continuations differs in the GPU check. Compact reload
preserves the live form exactly; source-faithful comparisons use the merged form.
