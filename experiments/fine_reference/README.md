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

## GPU availability correction, checked 2026-10-19

The authenticated quota endpoint reported a six-hour allowance already exceeded,
which we incorrectly interpreted as a hard execution blocker. The user's UI
showed 23h23m of 30h remaining. An actual GPU kernel subsequently completed on
**two Tesla T4s**, proving the earlier interpretation wrong. Treat the endpoint
as conflicting telemetry; verify allocation with an actual job before claiming
quota prevents execution. The formerly stated October 24 wait is retracted.

The two-case reference smoke job was submitted at
https://www.kaggle.com/code/shlydv/neuronpatch-fine-reference-smoke-20261019
(version 2). Version 1 failed before editing because the Kaggle Python3.13
host could not install tokenizer0.19.1. Version 2 creates an isolated Python3.11
environment and installs published torch2.0.1. Submission is not completion. See `gpu_allocation_check.json` and
`smoke_submission.json` for allocation evidence and source hashes. Next: inspect
startup, actual edits, and restoration; run the frozen 16-case comparison after
the smoke test succeeds. Existing custom QA experiments remain separate.

## Official metric audit

The harness now loads the exact `test_prediction_acc` and left/right slicing
functions from the pinned upstream evaluator using AST extraction, verified by
SHA256. Function bodies are unchanged; unrelated training/model dependencies
are not imported. The original code runs on CUDA during the real experiment.
CPU test doubles only exercise the evaluator's indexing and scoring locally.

- Rewrite/rephrase/portability: teacher-forced per-token accuracy, not free
  generation success or complete-answer exact match.
- Locality: agreement with pre-edit token predictions under the same supplied
  answer prefix. Preserving a wrong prediction can score 100% locality.
- The original driver repeatedly chooses the first nested answer alias, rather
  than taking the best score over valid aliases. The harness follows that rule.
- Knowledge editing prepends an EOS to GPT-J inputs, while the official metric
  tokenizes the prompt normally. These are distinct paths in upstream; retained.
- Rephrase generation and explicit greedy output inspection are supplemental;
  this wrapper does not reproduce every driver aggregation or paper table.

`check_metrics.py` checks singleton and variable-length batches, both padding
sides, causal answer slicing, partial token correctness, locality prediction
output, and first-alias selection. All passed on CPU. This validates metric
plumbing, not published model performance. Raw per-probe outputs and scores are
saved together to make disagreement visible.

## Kaggle packaging

`build_kaggle.py --output /tmp/fine-kaggle --cases 2` creates a private Kaggle
script bundle from the checked-out harness sources. It provisions Python3.11
rather than inheriting the Python3.13 host. It pins torch2.0.1, transformers4.40,
tokenizers0.19.1, huggingface-hub0.23.5, accelerate0.30.1 and numpy1.26.4.
NumPy and Accelerate are compatibility choices, not exact full-requirements
reproduction. It runs the CPU preflight before loading the actual GPU model.
The active version2 bootstrap predates this reusable builder but embeds the same
harness files and provisions the same environment; source hashes are recorded.
