# Broad MLP LoRA learning curve

Generated from saved checkpoints; exploratory development results only.

| Rate | Epoch | Parameters | Train supported EM /64 | Train correct refusals /64 | Anchors /20 | Val supported EM /16 | Val correct refusals /16 | Val refusals on supported /16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 1 | 829440 | 19 | 53 | 17 | 6 | 7 | 4 |
| 0.0001 | 2 | 829440 | 23 | 61 | 19 | 3 | 9 | 8 |
| 0.0001 | 3 | 829440 | 28 | 62 | 20 | 4 | 9 | 7 |
| 0.0003 | 1 | 829440 | 24 | 46 | 19 | 5 | 5 | 4 |
| 0.0003 | 2 | 829440 | 31 | 64 | 19 | 3 | 11 | 12 |
| 0.0003 | 3 | 829440 | 52 | 64 | 18 | 3 | 10 | 8 |

All-MLP rank-2 LoRA uses more parameters and potentially more steps than the sparse pilot. This is a learnability reference, not a matched-budget comparison. Exact match includes wording errors; inspect raw outputs before attributing gains to reasoning. Confirmation remains untouched.

Recorded checkpoints: 6/6. Completion must also be verified from the live process; checkpoint count alone is not process status.
