# neuronpatch

Train compact updates to selected neuron weights, evaluate what changes,
and export a patch tied to its exact base checkpoint.

**Status: experimental foundation.** Version 0.1 implements a tested PyTorch
weight-update primitive, not a validated behavioral editing system. Current experiments
expose correction/preservation trade-offs. Our custom QA experiments have not shown
a guided-selection advantage. A [16-case upstream FiNE reference](experiments/fine_reference/reference16_v1/REPORT.md)
gets the requested prefix on 15/16 original prompts versus 0/16 for random neurons,
but only 5/16 rewordings. We have not demonstrated an improved method or a
recommended recipe. See the [evidence ledger](docs/method-design.md).

The [first protected-row Qwen3-4B comparison](experiments/protected_rows/colab_v1/REPORT.md)
is complete: protection shows a partial correction/preservation trade-off improvement,
but all eight settings fail the declared joint success criteria. Output review finds
real unnecessary refusals as well as noisy exact-match labels. The experimental
constraint is not part of the public API or a recommended behavior recipe.
The [dose/bias diagnostic](experiments/protected_rows/dose_v1/REPORT.md) verifies
exported generations exactly and finds no measured evidence-discrimination gain;
part of the improvement comes from shifting the refusal trade-off.

The [external faithfulness probe](experiments/faithfulness_probe/colab_v1/REPORT.md)
finds real reference-answer interference, plus scoring and source-label problems.
Ignoring the untrusted passage is already a strong baseline in its factual mode.
No new trained method is promoted by that diagnostic.

A [two-case GPT-J transfer pilot](experiments/fine_transfer/pilot_v1/REPORT.md)
finds a positive selection effect: cross-prompt update gradients yield 6/6 requested
answer prefixes on untrained wordings versus 3/6 for FiNE supports given the same
training data. Its [frozen subject-scope follow-up](experiments/fine_transfer/scope_v1/REPORT.md)
then finds the requested target spreading to 9/16 other subjects, including all
eight people in the citizenship case. Original FiNE also leaks on 10/16 probes.
The transfer gate passes, but subject specificity fails. No current experimental
selector is recommended as a behavior-editing recipe.

The [fixed-support preservation experiment](experiments/subject_binding/pilot_v1/REPORT.md)
reduces unseen-subject leakage from 8/8 to 4/8 while retaining 6/6 requested
untrained-wording prefixes. Calibration succeeds, but the declared scope gate
fails and full continuations expose additional interference. This is a useful
partial result, not a validated adaptation recipe or superiority claim.

The completed [common-objective comparison](experiments/adapter_comparison/pilot_v2/REPORT.md)
favors broad official LoRA: 6/6 untrained-wording starts and 0/8 replacement
leaks, versus 5/6 and 1/8 for simpler mean-gradient sparse support, and 6/6 and
4/8 for saved consensus support. LoRA uses 50–200 times more trainable scalars;
this is not a parameter-matched efficiency result. All 530 capped generations
are verified and reviewed. Correct prefixes still hide later errors, and the
weak base makes utility retention an insufficient quality check. These are two
exposed development facts, not a validated recipe. The
[complete frozen-adapter group diagnostic](experiments/adapter_diagnosis/complete_v1/REPORT.md)
verifies18 interventions and1060 outputs without refitting or consuming fresh
facts. Paired MLP updates preserve6/6 requested starts but still produce bad
tails. Final-three-only updates give0/6; excluding them retains6/6. This narrows
the next design, not a validated recipe.

The [experimental paired-MLP implementation](experiments/paired_mlp/README.md)
now supports matching incoming rows and outgoing columns with FP32 deltas,
complete-response supervision through EOS, compact reload and merge checks.
This prepares a comparison at similar parameter counts; no pretrained behavioral
advantage has been demonstrated for it.

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
   experiments with pinned pretrained Qwen2.5-0.5B-Instruct. Behavioral robustness
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
