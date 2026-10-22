# The observed failures already exist before merging

The frozen replay completes with **zero new fits**. All **248 live-versus-merged
answer pairs are exactly identical**, including the broken high-rate responses,
the selected low-rate patch's unrelated Lakers target leakage, and its oxygen
symbol error. The deployment merge does not cause those observed failures.
This resolves the pending uncertainty for these eight endpoints, not for every
possible model, precision or input.

## Complete results

Each row aggregates both exposed development facts. The same saved patch is
evaluated live, then after its original FP16 merge. All existing target, scope
and utility test questions are included; no new questions or targets are added.

| Frozen condition | Requested transfer answers, live /8 | Merged /8 | Scored preservation losses, live /49 | Merged /49 | Changed live/merged answers /62 |
|---|---:|---:|---:|---:|---:|
| Gradient-paired, lr0.0003 |8|8|4|4|0|
| Gradient-paired, lr0.001 |5|5|16|16|0|
| Random-paired seed11, lr0.0003 |8|8|0|0|0|
| Official MLP LoRA, lr0.0003 |8|8|1|1|0|

These remain the original frozen exact-answer scores. Two of the selected
low-rate paired condition's four losses, and the selected LoRA condition's
single loss, are valid full formal UK names missing from the declared aliases.
They are not factual errors. The Lakers and oxygen-symbol errors are genuine
and present in both states. Literal target scans also miss other training-answer
spillover; see the [objective audit](../objective_diagnosis_v1.json).

Across all eight endpoints, merging changes zero top predictions at296 measured
teacher-forced response positions. Mean teacher KL(live||merged) ranges from
about8.76e-10 to1.08e-6, although some vocabulary logits move by up to0.128.
The logits are not bit-identical. Identical answers on this finite set are not
a universal numerical-equivalence guarantee.

## What explains the earlier high-rate discrepancy

Case0's final mean correction NLL is2.716726 in the live patch and2.718368 after
merging. Thus its poor endpoint already exists in the saved training patch.
Changing the merge arithmetic is not an evidenced remedy for this observed gap.

Prompt identity alone also does not explain it. For the exact correction prompt
sampled on the last training step, the recorded pre-update mean NLL is2.91256e-5,
whereas the saved live endpoint has mean NLL1.196853. The
[timing audit](timing_audit.json) binds that comparison to the saved tensors.
The optimizer update and gradient-enabled versus no-grad execution are still
different factors; the old worker did not save intervening checkpoints. This
experiment does not isolate their individual effects or prove a momentum cause.

The practical next test is to measure complete post-update endpoints and compare
an earliest-adequate checkpoint with the fixed50-step result. It should explicitly
compare gradient-enabled and no-grad evaluations at the same checkpoint, while
giving random supports and LoRA the same stopping/protection procedure. No new
training comparison is launched in this report. Stopping and regularization have
prior art, and their benefit here remains unmeasured.

## Review and verification

- [Kaggle version1](https://www.kaggle.com/code/shlydv/neuronpatch-paired-endpoint-v1-20261022)
  completes with return0 after1465.10seconds (24.4minutes). See
  [provider](provider.json) and [completion](completion.json). No optimizer was
  constructed; the worker reuses the eight exact saved patches.
- The [independent verifier](integrity.json) checks496 decoded observations,
  all16 teacher records, complete-response losses and both KL directions. Every
  merged teacher tensor and answer exactly matches the original parent. The
  worker also restores its original fingerprint and checks two base-generation
  controls after every endpoint; those16 controls are additional to496 scored
  observations.
- The accepted113726-byte source matches the frozen upload, SHA256
  `ed3c3750065ea1c927fcea0fe9394eb62263f6a868b443f0f1eb31e0aa6cc6e3`.
  All29 bundled file hashes, the downloaded submission, and the separate JSON
  download match. The worker verifies39 required parent hashes before loading
  the model. See [source integrity](source_integrity.json).
- [Review coverage](review.json) covers71 distinct prompt/reference/output
  contexts. Every context exactly matches a previously reviewed context, and
  all71 are reread here. This is assistant review, not blinded human assessment.
  Raw generated personal text and model tensors remain outside this public report.
- The independent verifier was subsequently tightened to absolute tolerances
  1e-10 for teacher metrics and1e-12 for divergence. All records still pass;
  the actual CPU worker/replay/analysis/review tests pass in FP32 and FP16.
  These postprocessing changes do not alter the accepted worker or protocol.

Two deliberately chosen development facts, with correlated wordings, cannot
establish a generally superior selection method. Fresh confirmation remains
unused. The broader useful-library and improved-method goal remains open.
