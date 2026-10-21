# Paired neuron updates with complete-response supervision

**Implementation and tokenizer checks only. No pretrained behavioral result.**
The [frozen GPT-J finisher](../adapter_diagnosis/complete_v1/REPORT.md) is now
complete, independently verified and consolidated. Its source and outcomes are
independent of this new implementation. No paired pretrained experiment has
started yet.

The [completed projection-family comparisons](../adapter_diagnosis/partial_v1/REPORT.md)
motivate testing incoming and outgoing updates together. They do not show that
pruning a trained LoRA adapter gives a good editor: retained answer starts can
hide false continuations. This implementation makes the next train-from-zero
comparison possible without asserting that it will succeed.

## What is implemented

`neuronpatch.paired.PairedMLPPatch` takes explicitly supplied MLP projection
paths and intermediate indices. For an ordinary MLP it trains matching incoming
rows and outgoing columns. For a gated MLP it trains both gate/up rows plus
matching down columns. Biases and every other original parameter stay frozen.
This combines parameter access studied by incoming-row and outgoing-column
methods; pairing alone is not a novelty claim or a reproduction of their methods.

New deltas default to FP32 independently of the base dtype. The implementation
uses ordinary child modules, not generation hooks. Compact save/load checks the
original base fingerprint; merge adds in promoted precision and casts once to
the base dtype. Low-precision live and merged outputs can still differ. The
class is experimental and is not exported from the package's top-level API.

```python
from neuronpatch.paired import MLPSelection, PairedMLPPatch

patch = PairedMLPPatch(
    model,
    [
        MLPSelection(
            incoming=("model.layers.12.mlp.gate_proj", "model.layers.12.mlp.up_proj"),
            outgoing="model.layers.12.mlp.down_proj",
            indices=(14, 98),
        )
    ],
)
# These indices illustrate the API; they are not a selected/recommended support.
```

For hidden width h and k selected intermediate indices, the gated version
trains **3kh scalar weights**, while the ordinary version trains **2kh**.
Shapes do not prove that the paths belong to the same computational MLP; callers
must verify the architecture. Only ordinary materialized floating-point Linear
projections are supported. Quantized/fused projections, tied targets, concurrent
patch composition and distributed training are outside this implementation.

`neuronpatch.supervision.chat_response` constructs a complete teacher-forced
assistant response from the pinned chat template. It checks an exact token
prefix, includes every response token through EOS, rejects ambiguous boundaries,
and refuses to truncate. It explicitly requests token lists, including under
Transformers5, whose default template result is a dictionary-like object.

`neuronpatch.losses.completion_nll` applies the causal shift once, trains the
selected response tokens including EOS, and averages tokens within each example
before averaging examples. Prompt/padding logits receive no direct loss. This
is ordinary SFT; it does not guarantee factual or free-running response quality.

The experimental selection helper captures all-position local gradient factors
of that same full-response loss on a frozen base. It ranks the norm of the mean
gradient for the paired rows/column, using CPU Gram products instead of dense
model-sized weight gradients. Examples receive equal weight; opposing gradients
cancel before ranking. A causal toy checks scores against actual dense backward
gradients, including prompt-position effects. This is a simple group-gradient
baseline, not a claim that gradient magnitude predicts selective generalization.

```python
from neuronpatch.losses import completion_nll
from neuronpatch.supervision import chat_response

example = chat_response(tokenizer, messages, desired_complete_answer, max_length=512)
output = model(input_ids=example.input_ids, attention_mask=example.attention_mask)
loss = completion_nll(
    output.logits,
    example.input_ids,
    example.completion_mask,
    example.attention_mask,
)
loss.backward()
# optimizer.step(), zero_grad(), and preservation examples are caller-owned.
```

## Checks and their limits

- Independent dense nonlinear references verify both ordinary/gated forward
  computations, input/delta gradients and exact unchanged weight coordinates.
- FP16-base/FP32-delta training is exercised under CPU autocast. Merged weights
  match an independently constructed promoted-addition reference exactly.
- Compact reload matches live predictions. Wrong bases, altered indices,
  nonfinite deltas, shared targets and invalid requests are rejected.
- A randomly initialized tiny Qwen3 performs an actual complete-response update,
  then reloads and merges. This is a numerical integration check, not language
  understanding or selective generalization.
- The real pinned Qwen tokenizer checks short factual, abstention, numeric and
  Hindi responses, retaining the exact text and the intended stop token. No
  pretrained model weights are downloaded or executed by that preflight.

```bash
pytest -q tests/test_paired.py tests/test_completion_loss.py tests/test_supervision.py
python -m experiments.paired_mlp.preflight --output artifacts/paired-preflight.json
```

The [tokenizer/config record](preflight_v1.json) pins
`Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, a non-thinking instruction model
described in its [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
This is the same instruction-model revision used in earlier QA work, not an
unexplored stronger model or a guarantee that the earlier task will now work.

## Next comparison, not yet registered or launched

First use exposed development facts to check train-from-zero correction,
subject specificity, response stopping and unrelated answers on the instruction
model. Do not spend the prepared fresh bank merely to debug the new objective.
Use complete correction **and preservation** answers, not a country prefix plus
four weak-base prediction contexts. Base-answer checks must establish whether
the preservation task has meaningful initial accuracy. A truncated generation
is incomplete, never a complete-answer pass.

Compare paired gradient-selected supports with paired matched random supports,
outgoing-only supports at the same scalar count, and small official LoRA over
the same MLP layers. Random supports need multiple seeds. Discovery compute,
optimizer steps, trainable scalars and tuning data must be reported separately.
Selection and fitting must not inspect evaluation outputs.

Fair parameter budgets are feasible: from the pinned configuration, rank-one
gate/up/down LoRA across 36 layers has 1,327,104 scalars. A total of 173 paired
intermediate indices has 1,328,640 scalars, **0.116%** more, and 519 outgoing-only
indices has exactly the latter count. This is count arithmetic only. Layer
allocation, selector, learning rates, data and scoring still need a frozen
execution specification. It does not imply equal expressiveness or compute.

The useful result would be better complete-answer correction at comparable
preservation and budget, or the same correction with demonstrably less data or
storage, consistently against those controls on fresh cases. If the paired
representation offers no advantage, report that and avoid another round of
selector complexity without evidence. We have not yet earned a default recipe,
paper-superiority claim or a new algorithm claim.
