# FiNE reference comparison — prepared, not reproduced

Purpose: establish an upstream reference before interpreting NeuronPatch's
selection failures. This is a diagnostic step toward the library goal, not a new
method or a replacement for the goal.

## Why this reference

The [official source](https://github.com/opanhw/fine-unified-editing-code)
contains both the knowledge editor and a random-selection branch. We pin commit
`a2489cbac0dfe5a956385734065cfa041384e7d1`, GPT-J-6B revision
`47e169305d2e8376be1d31e765533382721b2cc1`, and KnowEdit CounterFact revision
`6d218881e60c7c8be836bd2befd598eb3b318be4`.
The benchmark contains 839 cases. The initial comparison uses its first 16,
selected without looking at model outcomes. It is not a full-table replication.

The source configuration uses five neuron selections per target token, excludes
the final three layers, and permits up to 50 Adam steps at 0.001. Its objective,
stopping condition, activation scoring, BF16 deltas, and random branch remain
unchanged. The wrapper replaces only the model's placeholder path and imports
modules directly to avoid unrelated EasyEdit dependencies. It restores the base
weights after each individual edit, checking the baseline generation afterward.
Upstream resets the seed to 42; this initial run is not a multi-seed estimate.
Random supports are drawn with replacement in upstream; do not silently fix that.

## Checked locally

`preflight.json` records a CPU check using a tiny random GPT-J:

- unchanged upstream imports and published hyperparameters load;
- upstream and standard GPT-J with identical weights have identical logits;
- captured MLP activations have the expected count and dimensions.

This does **not** test CUDA editing, large-model memory use, convergence, the
random-versus-selected effect, or complete paper evaluation. Local torch differs
from the upstream pinned environment and is recorded explicitly.

## Run

Use a separate CUDA environment with transformers 4.40.0, tokenizers 0.19.1,
huggingface-hub 0.23.5, torch, accelerate, numpy, PyYAML, requests, and tqdm.
The original requirements pin torch 2.0.1; prefer that compatible environment
for the reference and document any unavoidable runtime differences. Do not
replace the model with the locally cached 0.5B Qwen or quantize it silently.

```sh
git clone https://github.com/opanhw/fine-unified-editing-code /tmp/fine-reference
git -C /tmp/fine-reference checkout a2489cbac0dfe5a956385734065cfa041384e7d1
python experiments/fine_reference/run.py --upstream /tmp/fine-reference --output /tmp/fine-preflight --preflight
python experiments/fine_reference/run.py --upstream /tmp/fine-reference --output /tmp/fine-results --cases 16
```

The second command requires CUDA. Outputs must go to a fresh directory.
The benchmark is downloaded from a pinned revision and SHA256-verified.
Model download is approximately a multi-GB full GPT-J checkpoint; GPU capacity
must accommodate model weights, edit gradients, activations, and saved originals.

The wrapper records raw greedy continuations for rewrite, rephrase, locality,
and portability prompts. These are supplementary output diagnostics. **Before
claiming numerical reproduction of a paper table, run/audit the original
benchmark evaluator as well**: generated strings are not interchangeable with
teacher-forced token accuracy or prediction-agreement locality. Review actual
outputs alongside those metrics. A bounded success means editing operates and
selection has a measurable advantage under a specified comparable metric; no
advantage is an outcome to investigate, not a reason to change cases afterward.

## Current execution blocker, checked 2026-10-19

Kaggle's authenticated quota API reported GPU used 23,767.22 seconds, allowance
21,600 seconds, reserved zero, pay-to-scale disabled. Refresh listed as
2026-10-24 00:00 UTC (05:30 India time). No new GPU run was submitted.
TPU availability does not make the CUDA-specific source runnable unchanged.

Next: CUDA environment check and two-case smoke run, then the frozen 16-case
comparison if the smoke run succeeds; audit official metrics before expanding
or interpreting superiority. Existing custom QA experiments remain separate.
