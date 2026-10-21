# Fixed-support subject binding ablation

The [private Kaggle run](https://www.kaggle.com/code/shlydv/neuronpatch-subject-binding-v1-20261020)
completed as version 1, kernel 139969733, return 0 on 2026-10-21 UTC. The source
was independently pulled back and matches the frozen 222066-byte upload exactly.
The [verified report](pilot_v1/REPORT.md) finds a partial improvement: target
prefix transfer remains 6/6, unseen-subject leakage falls 8/8→4/8, but the declared
scope gate fails. All 424 generations, six calibration records, eight cold
sequences and the final base fingerprint verify. The slug and
protocol retain their original development identifier. See [launch metadata](launch_v1.json).

The [completed scope check](../fine_transfer/scope_v1/REPORT.md) finds that the
cross-prompt-gradient edit transfers its target to 9/16 other subjects. This
development experiment tests whether changing the preservation objective can
repair that failure while keeping the selected neurons fixed. It is not a new
selector benchmark, fresh-fact confirmation or a recommended library recipe.

## Frozen comparison

- Two exposed GPT-J-6B edits, original three correction phrases and exact saved
  consensus supports. Each new fit starts with zero deltas; there is no warm start,
  new discovery, support change or strength scaling.
- Preservation weights 0,1,10: six fits, each Adam lr .001 for exactly 50 updates.
  The unprotected control also gets 50 steps, differing from the earlier pilot's
  early stopping. Compare against this matched control, not just the old patch.
- Existing worst token/prompt NLL, prefix KL, repetition and global delta norm.
  The added loss preserves full-vocabulary base predictions at the first up to
  four answer positions of three different subjects per edit. Base greedy tokens
  provide the teacher-forced contexts; no benchmark answer label is trained.
  Average positions within each prompt, then average the three prompts equally.
- Calibration subjects use the first exact-template matches at source indices
  256..383. Four new evaluation subjects per edit use 384..511. Exclude the previous
  scope probes, both edited subjects and all prepared fresh-bank subjects.
  The already exposed 16 scope probes and 32 global utility prompts are separate
  secondary checks. No model-output filtering; indices>=512 remain uninferred.

The actual merged FP16 model determines final calibration statistics and generated
evaluation, rather than assuming live BF16-delta hooks round identically. Retain
live metrics separately. Save calibration logits and target-token log probabilities
so an independent verifier can recompute CPU float64 KL and model selection.

## Selection and interpretation

Before scoring generated evaluation outputs, choose the smallest positive weight
whose **both** cases have final maximum target NLL <= -log(.9) and maximum
calibration-token KL <= .1. No qualifying weight means no selected candidate. Do
not pick a different weight because its evaluation generations look better.
All three weights and both cases are reported regardless of selection.

A selected candidate's count gate requires 2/2 original prefixes, at least 5/6
untrained-wording prefixes and at most one fewer than the fixed 50 control;
new-subject target leakage at most 1/8, strictly lower than control and at most
half its count; no additional lost reference-prefix or global-utility answers.
Loss sets are compared per probe rather than only as equal aggregate totals.
Full outputs still require review. Empty baseline-correct sets, ambiguous labels,
alias mismatches and the 40-token cap limit conclusions. Passing these counts
alone cannot establish a usable recipe, data efficiency or superiority to papers.

Prediction: calibration leakage should fall with answer preservation; transfer
to unseen subjects is uncertain. If calibration improves but held-out scope does
not, finite anchors did not generalize. If target learning collapses, the fixed
supports may lack the required capacity, or the optimization trade-off may be
too severe. Neither result establishes a universal impossibility.

## Execution and verification

One private Kaggle worker, 90-minute worker limit and 95-minute outer timeout;
no overlapping worker. The pinned reference base/runtime and two saved patches
must reproduce all 8 cold sequences before new inference. Every fit must restore
base generation, and the final full checkpoint fingerprint must match.

The CPU tests verify the answer-KL gradient, exact reduction to the unprotected
control, accumulated versus joint gradients, calibration-only penalty selection,
and the complete tiny-model six-fit worker including exported artifacts and
independent calibration reconstruction. They do not validate full-size behavior.
Strong mean-gradient/low-rank comparisons and fresh cases remain necessary after
this objective diagnosis; they are not silently replaced by this smaller test.

```sh
python -m experiments.subject_binding.prepare \
  --source /path/to/pinned-test_cf.json \
  --pilot-results /path/to/transfer-results/results --output /tmp/binding-data
python -m experiments.subject_binding.build \
  --data /tmp/binding-data --output /tmp/binding-bundle
# After execution and download, verify before scoring:
python -m experiments.subject_binding.verify \
  --download /path/to/binding-results --executed-source /path/to/pulled-script.py \
  --decode --output /tmp/binding-integrity.json
python -m experiments.subject_binding.analysis \
  --results /path/to/binding-results/results --output /tmp/binding-analysis
```

The independent decode requires the cached pinned Transformers 4.40 tokenizer.
Raw generated text remains in local artifacts; do not republish fabricated
personal allegations as examples. Public decisions must disclose omitted text
and the resulting limits on independent rescoring.

Independent postprocessing also checks the actual recorded Torch version and
target-token lengths. Utility reporting includes total prefix hits and exact
continuation changes, alongside lost baseline-correct answers; changed text is
not uniformly damage. These additions do not change the submitted worker,
penalty selection, labels or advancement criteria. The selector's AST was
checked identical to the frozen upload before reading new evaluation outputs.

`review.py` builds a private packet covering all 424 saved continuations. It
deduplicates only identical question/reference/output contexts, retaining every
condition occurrence. An identical answer to a different question remains a
separate review item. Review all distinct contexts before interpreting complete
answer quality. Plot learning/calibration separately from unseen-subject results:

```sh
python -m experiments.subject_binding.review \
  --results /path/to/binding-results/results --output /tmp/private-review.json
python -m experiments.subject_binding.plot \
  --fits /path/to/binding-results/results/fits.json --output /tmp/binding-fit.png
```

Figure commands additionally require Matplotlib; the core library does not.
The completed report includes a separate leakage figure so good calibration
curves cannot be mistaken for successful unseen-subject preservation.
