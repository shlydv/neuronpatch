# Isolate preservation coverage before changing localization

This is an exploratory mechanism test, not a final evaluation or a new-method
claim. Run after the existing 2048 size curve completes; do not interrupt it.

Use the same pinned Qwen model and the existing 512-example training prefix,
256-example development set, all-MLP rank-2 LoRA, seed, supervised order,
256 optimizer steps, two replay examples per step, coefficient 1 and both learning
rates (.0001/.0003). Preserve the response-token KL implementation and reference
model. Keep confirmation unused. Do not alter targets or prompt format.

The only treatment is anchor eligibility: the original 91 exact-correct training
examples plus 24 reviewed valid nonexact examples from manual_review_512.json.
Use the original gold continuation for teacher-forced KL, as the control does;
do not switch to generated continuations or preserve trailing copied text.
Shuffle the enlarged pool with the same seed and cycle it using the existing
policy. This redistributes a fixed 512 replay-example exposures across 115 rather
than 91 anchors. Report that dilution and actual per-example exposure counts.
The existing 512 fits are the controls; verify source/configuration parity.

The prepared review manifest now binds each judgment to an SHA-256 digest of its
passage, question, aliases and baseline output. `reviewed_anchors.py` validates
training membership, evidence identity, decision/list consistency and nonexact
nonrefusal eligibility before returning additions. Three focused tests exercise
leakage, changed evidence, rejected decisions and invalid additions. This helper
is integrated through the runner's optional `--review-manifest` argument. The
already-running size curve continues executing its loaded original code; that
source was archived and checked against its recorded hash before this edit.
The validated pools contain 91 and 115 unique examples. At 512 replay exposures,
the control gives 57 anchors six exposures and 34 five; expansion gives 52 five
and 63 four. This budget dilution is part of the treatment, not hidden extra compute.

Prepared invocation (not yet launched):

```sh
HF_HOME=/private/tmp/neuronpatch-hf HF_HUB_OFFLINE=1 .venv/bin/python experiments/real_qa/run.py --scaled-size 512 --review-manifest experiments/real_qa/anchor_coverage/manual_review_512.json
```

Results go to a new `coverage_512` directory; existing experiments cannot be
overwritten. The runner records review hashes, exact/added IDs, executed sources
and per-fit replay counts. Default training logic, targets and optimizer are
unchanged. The full two-rate size curve remains the prerequisite for launch.

Primary mechanism observation: how many of the 24 added examples remain valid
answers, become refusals, become wrong answers, or merely change wording? Inspect
all 24, including apparent regressions and improvements. Separately track the 91
original anchors to reveal protection lost through dilution. Record raw EM/F1,
refusals, output pairs and the same qualitative review categories for both arms.

Development utility is necessary: an improvement confined to the protected
training answers is insufficient. Compare supported answers, correct missing-
evidence refusals, retained base-correct answers, and semantic output changes on
identical development questions. More retained training answers without improved
development trade-offs is a coverage fix, not evidence for a better library method.
No effect or worse trade-offs means coverage alone is insufficient. Investigate
those outputs rather than retroactively broadening the success claim.

Do not bundle decision-margin penalties, a new selector, clustering, or an
additional data change into this test. If coverage helps, carry the same policy
into matched random-neuron and guided-neuron comparisons so any selection claim
must add value beyond a better training objective. If it does not, the next
diagnostic is whether distribution preservation misses small but decisive changes
in answer/refusal probability—not another activity-based score by default.
