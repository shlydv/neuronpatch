# No-training dose and refusal-bias diagnostic

Exploratory exposed-pilot diagnostic; no training or fresh confirmation.

Calibration protected mean UNKNOWN log-odds shift: 15.44605.
Matched unprotected scale: 0.47603; relative signal mismatch: 0.60%.
Cold-loaded saved patches exactly reproduce all 320 validation/utility trajectories.

| Condition | Supported /64 | Missing /64 | Old correct retained /38 | Utility /32 | Primary pass |
|---|---:|---:|---:|---:|---|
| protected_full | 36 | 53 | 29 | 31 | False |
| unprotected_full | 23 | 57 | 19 | 32 | False |
| unprotected_matched | 37 | 50 | 32 | 32 | False |
| unknown_bias | 35 | 48 | 35 | 31 | False |

Secondary first-token missing-label AUROC: base 0.9080; protected_full 0.8938; unprotected_full 0.8936; unprotected_matched 0.8970; unknown_bias 0.9080. This uses the same imperfect labels and is not a primary promotion metric.

The mean matches one refusal signal on calibration, not the full output distribution. The constant-bias control only changes the first generated token. Secondary assistant labels are diagnostic and do not alter the frozen evaluator.
