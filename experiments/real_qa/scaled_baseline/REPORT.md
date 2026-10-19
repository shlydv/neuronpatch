# Expanded unedited-model baseline

Pinned Qwen2.5-0.5B-Instruct, unchanged prompt/scorer/greedy generation, 2026-10-19.
All 192 previously evaluated train/calibration/validation outputs reproduce exactly.
No confirmation inference. The additional validation rows are now development data.

| Split | Supported EM | Supported F1 | Supported refusals | Correct impossible refusals |
|---|---:|---:|---:|---:|
| Training | 360/1024 | .501 | 118/1024 | 315/1024 |
| Calibration | 55/128 | .572 | 11/128 | 40/128 |
| Validation | 45/128 | .458 | 13/128 | 34/128 |

On the 224 additional validation rows alone, supported EM is 41/112, supported F1
is .457, supported refusals are 12/112, and correct impossible refusals are 31/112.
The original 16-pair pilot had different rates; do not compare a small old subset
to a larger new set and attribute the difference to an adapter. Every data-size fit
must be evaluated on the same expanded set, with old/new subsets reported separately.

Exact-correct supported preservation anchors number 20, 91 and 360 for the nested
128/512/2048 training prefixes. Anchor selection validates source question/context/
answers against the corresponding baseline and excludes out-of-prefix examples.
This remains a conservative exact-match policy; it omits some semantically correct
paraphrases. No validation predictions are used to choose anchors.

These are extraction metrics, not semantic accuracy. Raw predictions, token-limit
flags, model/source/data hashes, and new-only metrics are retained. Thirty-six training, six calibration
and three validation outputs hit the 32-token cap; none is silently discarded.
No adapted-model claim follows from these baseline measurements.
