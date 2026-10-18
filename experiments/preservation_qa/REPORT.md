# Preservation follow-up: random controls pass; both guided selectors fail

Completed 2026-10-19 on the same pinned pretrained Qwen2.5-0.5B-Instruct checkpoint.
[Protocol](PROTOCOL.md) was fixed before outcomes. This is a synthetic extraction
pilot, not evidence of broad model-editing robustness.

## What went wrong previously

The old patch increased the UNKNOWN-versus-correct-first-token logit margin on all
24 supported questions. The mean increase was 7.84 on the 17 retained answers and
14.84 on the seven broken ones. For all seven failures, this margin crossed from
negative to positive. This establishes a broad bias toward abstention in this test,
not a complete causal circuit explanation. It is a first-token comparison, not the
probability of the whole answer. See [raw diagnostic](previous_diagnostic.json) and
`diagnose_previous.py`. These previously exposed examples were not tuning data.

## Fresh held-out results

| Method selected using validation only | Supported / 48 | Correct abstentions / 48 | Total / 96 | Final preservation gate |
|---|---:|---:|---:|---|
| Prompted base | 48 | 24 | 72 | Reference |
| Activation contrast, lr 0.0003 | 41 | 46 | 87 | Fail |
| Gradient sensitivity ratio, lr 0.0003 | 40 | 48 | 88 | Fail |
| LoRA, lr 0.001 | 44 | 48 | 92 | Fail |
| Random seed 11, lr 0.001 | 48 | 42 | 90 | Pass |
| Random seed 29, lr 0.001 | 48 | 42 | 90 | Pass |
| Random seed 47, lr 0.0003 | 48 | 40 | 88 | Pass |

All methods scored 16/16 on the arithmetic sentinel, which was also used in
validation and therefore is not an untouched capability benchmark. Every chosen
patch had 48/48 validation QA accuracy, including both guided selectors. No
candidate required the no-edit fallback on validation. The held-out family shift
still exposed substantial damage. A validation gate is useful rejection machinery,
not a guarantee that preservation will generalize.

The new selector's 91.7% balanced accuracy was only 1.0 point above the original
selector and 1.4 points below the random-control mean (93.1%). It failed both the
preservation and superiority gates. LoRA scored 95.8%, but still lost 4/48 supported
answers, exceeding the preset 5-point tolerance. All three random runs met the
pilot's gate: +33.3 to +37.5 points correct abstention with no supported-answer loss.
Do not generalize these three seeds to reliable performance on real documents.

## What changed and what is supported

- The new gradient selector uses correction loss sensitivity relative to supported
  answer sensitivity. Squared gradient norms are averaged per example before
  ranking, avoiding cancellation. It overlaps the original selector on only 2, 1
  and 5 of 16 columns in the three layers, respectively. Changing the selected
  neurons substantially did not solve over-abstention.
- Every method, including every random seed, received the same two-rate tuning
  budget. The objective was unchanged, isolating the selection comparison within
  this experiment. Differences from the previous pilot also include data and
  learning-rate search; cross-pilot improvements cannot be attributed to one change.
- We added a tested validation policy with an explicit no-edit outcome. It remains
  experiment code because its thresholds and metrics are task-specific. We did not
  promote the unsuccessful gradient selector to a supported library API.
- The core library required no change for this follow-up. All selected-column
  patches were reloaded before final evaluation; fixed-input logits matched exactly.
- There are 24 passing local tests, including split isolation and rejection of
  aggregate improvements that damage supported answers.

Raw predictions, validation candidates, selected indices, gradient energies,
parameter counts, source/data hashes, timings and versions are in
[results/summary.json](results/summary.json). Columns train 43,008 scalar weights;
LoRA trains 34,560. Discovery took approximately 1.74 seconds here; this is not an
end-to-end performance claim and may not scale to larger models.

## Product decision

Keep random column selection as the honest baseline and both guided methods as
research options. Do not market guided selection as safer or more effective. A
useful library must expose the improvement/damage trade-off, retain a no-edit
option, and help users evaluate on document families unlike their training data.

The next justified milestone is a small real-document evaluation with explicit
supported and unsupported questions and an independent preservation set. Establish
whether the random baseline provides useful value there before adding another
selector. If optimizing again on this synthetic task, train against preserved
behavior explicitly (for example, compare preservation regularization with ordinary
SFT under matched budgets) and use new held-out data. This experiment stops here;
the now-exposed test set must not be presented as fresh in subsequent work.

## Reproduce

From repository root, install `pip install -e '.[dev,experiment]'` and run
`HF_HOME=/path/to/cache python experiments/preservation_qa/run.py` on an Apple GPU.
The results directory must not exist; move the published results aside first.
The runner deliberately fails rather than silently overwriting a previous run.
Model binaries and patch/probe binaries are gitignored. For the old diagnostic,
the original pilot's locally saved patch must also be present.
