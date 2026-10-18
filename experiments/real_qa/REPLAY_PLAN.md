# Next bounded objective test: repeated preservation anchors

Motivation from saved-adapter diagnostics: the low-rate selected patch turns three
of its 20 protected, originally correct training answers into UNKNOWN. A fourth
loses exact match through harmless wording. This already contradicts a pure
held-out-transfer explanation. The current KL is computed in only 18 of 64 steps:
anchors are protected when their row appears, but other updates can change them.

Hypothesis, not a conclusion: applying preservation throughout training may reduce
this damage without changing the selected neurons. It could instead suppress useful
learning or protect the anchors without transferring. Those outcomes are meaningful.

Keep the existing model, examples, selected supports, random seed, row count, LoRA
rank, supervised order and optimizer settings. Compare the same two learning rates
for selected rows, random rows and LoRA. Use a separate deterministic stream of two
reference-correct training anchors each step for KL(reference || edited), including
EOS. Keep supervised batches unchanged. Record extra forward-pass cost. This tests
the preservation schedule/exposure jointly; it is not an equal-compute comparison
or a claim that replay is a novel algorithm. No calibration/validation labels enter
anchor selection, and no confirmation inference is warranted yet.

Report anchor exact match and raw-output failures, supported/unsupported train and
validation behavior, and first-token margin shifts. Examine new refusals separately
from wording. Improvement on training anchors alone is insufficient; retained
corrections and behavior on unseen articles matter. If correction disappears when
anchors are protected, investigate gradient conflict and actual update suitability
instead of relaxing preservation or inventing another activation score.

This is an objective diagnostic, not the final method. Strong localization,
capacity sharing, paper-faithful baselines, independent confirmation and broader
models/tasks remain required to establish a useful combined approach.
