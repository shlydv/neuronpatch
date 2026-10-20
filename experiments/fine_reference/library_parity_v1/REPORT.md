# Ordinary-HF library workflow matches upstream on the 6B check

Kaggle version 1 completed on 2026-10-20 using GPT-J-6B. Both implementations ran
under the same torch 2.4.1+cu121 / Transformers 4.40.0 runtime, with the same pinned
checkpoint, FP16 base, BF16 deltas, source hyperparameters, seed and fixed inputs.
The full library revision and exact bundled file hashes are recorded in the
submission; those hashes are authoritative.

Two exposed smoke cases were tested with contribution and uniform-random supports.
This validates a faithful baseline implementation; it is not a new-method result.

| Case | Selection | Ordered supports | Learned deltas | Merged selected columns | Fixed merged generations |
|---|---|---|---|---|---|
|0|Contribution|Exact|Exact|Exact|Exact|
|0|Random|Exact|Exact|Exact|Exact|
|1|Contribution|Exact|Exact|Exact|Exact|
|1|Random|Exact|Exact|Exact|Exact|

Every learned-delta difference is zero. The compact artifacts have shapes
5×4096 for case 0 and 20×4096 for case 1. Downloaded safetensor deltas were also
independently compared with the upstream tensors on CPU: all four match exactly.
The four library artifacts have the same untouched base fingerprint.

All eight base-restoration checks pass. Compact save/reload preserves exact deltas
and live generations. The 18 merged generation outputs match the original editor;
these same 18 baseline/merged outputs also match the earlier torch 2.0.1 run.
That last check is a limited cross-runtime observation, not universal numerical
compatibility. See [summary.json](results/summary.json), the per-condition comparisons,
and [integrity.json](integrity.json).

## Live and merged precision differ

One of 18 live-hook continuations differs from its merged-weight continuation:
the random case 0 rephrasing changes its invented biography continuation. The
requested counterfactual target is missed in both forms. This is an observed
mixed-precision deployment difference, not a training-port discrepancy. Live
save/reload is exact, while the source-faithful deployment comparison uses merged
weights. Do not promise bit-identical live and merged outputs generally.

## Scope and provenance

- The library uses ordinary Hugging Face GPT-J and scoped hooks, without requiring
  the original modified model class.
- The actual full-size run meets every predeclared parity criterion. The unchanged
  upstream source, tokenizer/model/data revisions and all bundled library hashes
  are recorded in the submission and manifest.
- Total run time is about 1495 seconds/24.9 minutes, including two sequential model
  loads, source checks, repeated fingerprinting, reload checks and generation.
  It is not an editing-speed comparison.
- Binary edit/reference tensors remain in the downloaded local Kaggle outputs;
  their hashes and artifact metadata are saved here without committing weights.
- Version 1 copied initial source defaults into artifact metadata, leaving
  hparams.random=false even in random-condition artifacts. The condition is
  established by filename, comparison record, ordered supports and independently
  verified deltas. Raw artifacts are unchanged. The future harness now writes
  effective per-condition selection metadata; no training behavior was changed.

Transformers 4.40.0 is the explicit isolated reference runtime, below the optional
general integration extra's minimum. Current-library integration was separately
checked on a tiny ordinary GPT-J. This result does not validate every installed
Transformers version, architecture, device or checkpoint.

## Next quality work

The source-to-library gap is now closed for this bounded 6B configuration. New
work should test the proposed protected incoming-row updates against useful
generated behavior and held-out preservation. The 16-case reference shows the
actual transfer/interference weaknesses; the new prototype remains outside the
public API until a language-model comparison demonstrates a benefit.
