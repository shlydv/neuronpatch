# Broader development task: baseline qualification first

This frozen slice prepares a diagnostic, not a paper reproduction or a new
method claim. There are no model answers or training results yet.

The baseline was submitted as private Kaggle
`shlydv/neuronpatch-math-baseline-v1-20261023`, version 1, kernel 140420814.
It was observed RUNNING and the accepted source hash matches
`submission_frozen.json`. Provider/worker/internal limits are 3600/3000/2700
seconds. Submission and execution are not evidence of a completed result.

Source: [official GSM8K repository](https://github.com/openai/grade-school-math),
commit `3101c7d5072418e28b9008a6636bde82a006892c`. `source.json` records the
downloaded training-file hash and the bundled original MIT license. The official
test split was not downloaded. Training-set provenance does not imply the model
has never encountered these public problems in pretraining.

The deterministic partition reserves 128 training examples, 32 separate
calibration examples and 128 development examples. All are development data;
none is advertised as final confirmation. Normalized duplicate-question groups
are excluded before partitioning, and partition membership is independent of
source row order. Exact-text deduplication does not eliminate semantic near-duplicates.
Calculator annotations are removed using the official repository's documented
convention. The full worked solution and marked numeric answer remain.

The pinned Qwen3-4B tokenizer successfully checks all 288 complete examples under
a 1,024-token limit. The longest is 384 tokens; the longest reference completion
is 265 tokens. These reference lengths do not guarantee that model generations
fit the output budget.

## First run's decision

Run the unmodified pinned Qwen3-4B-Instruct-2507 on the 128 development examples
with the provided reasoning prompt and greedy decoding. Use at most 512 new
tokens per example. Preserve full responses, completion status and token IDs.
No model fitting belongs in this initial run.

The source references use a final `####` marker. The strict parser never treats
an arbitrary number in the reasoning as the final answer. Review unparsed
responses separately: a correct answer without the marker is a format miss,
not evidence that arithmetic adaptation is needed. Truncated answers likewise
do not count toward the qualification threshold.

At least sixteen completed, mathematically wrong answers among 128 are required
to justify using this slice for an adaptation diagnostic. This is a pragmatic
pilot gate, not a statistical power calculation or proof that those errors can
be learned. If the gate fails, report the ceiling/execution limit and do not
automatically train, shorten the prompt, or select a more convenient subset.

If it passes, freeze the subsequent comparison's optimization, calibration and
preservation banks before fitting. The next comparison must include LoRA and
random supports and measure actual-update prediction errors at fixed steps.
Unrelated-task preservation data and a full training protocol are
still to be prepared. The baseline worker and finite GPU caps are now frozen;
post-run verification and full response review remain required. Accurate
prediction alone would not demonstrate a useful adaptation advantage.

Rebuild from the pinned source with:

```sh
python -m experiments.relation_learning.prepare_math \
  --source artifacts/gsm8k_source/train.jsonl --output /tmp/math-development
```
