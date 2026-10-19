# The 128-example reference on common expanded validation

Re-evaluated both existing one-epoch, all-MLP LoRA checkpoints without retraining.
All original 32 validation outputs match. The added 224 rows are now development
examples, not final confirmation. Every larger training-size fit uses these same
256 validation rows.

| Model | Supported EM /128 | Supported refusals /128 | Correct impossible refusals /128 |
|---|---:|---:|---:|
| Base | 45 | 13 | 34 |
| 128 examples, lr .0001 | 40 | 47 | 81 |
| 128 examples, lr .0003 | 43 | 33 | 62 |

The broader sample confirms a correction/refusal trade-off. It does not establish
that any given lost exact match is a semantic failure, or that each new exact match
is a genuine correction. Read raw outputs, preserve F1 and known-correct retention,
and distinguish warranted from unnecessary refusals. The two rates are retained
rather than selecting one after inspecting this set. No selector is tested here.
