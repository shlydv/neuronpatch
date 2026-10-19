# Correct-answer protection omits useful training answers

The existing preservation rule uses exact match. A training-only audit queues
nonexact baseline answers containing a complete normalized reference span. This
is a review heuristic, not a semantic scorer or proposed automatic anchor policy.
It reads no validation or confirmation examples and changes no training inputs.

| Training prefix | Supported examples | Exact anchors | Additional review candidates |
|---|---:|---:|---:|
| 128 | 64 | 20 | 9 |
| 512 | 256 | 91 | 28 |
| 2048 | 1024 | 360 | 118 |

All nine candidates in the original training prefix were manually inspected with
their passages. Seven give valid answers with extra wording. One gives the right
date followed by a dangling excerpt. One incorrectly turns an argument about the
Council of Nicaea into a decision attributed to the Council. This last example
demonstrates why reference-span containment is not sufficient for preservation.
Annotations are assistant judgments, not blinded labels or a new benchmark.

Among the seven clearly valid omitted answers, the 512-example .0001 fit refuses
four and the .0003 fit refuses two. These examples received supervised answer
training but no explicit preservation replay. This establishes a coverage gap and
training-set damage; it does not prove adding replay will fix them. Exact-correct
anchors also suffered damage in prior experiments, so coverage is not the only
problem.

Across all 28 unreviewed candidates in the 512 prefix, the two fits refuse 19 and
14 respectively. Do not call these counts semantic regressions: correctness of
the full candidate set remains unreviewed. The seven reviewed cases above provide
the narrower supported claim. The queue stores full outputs/contexts and source
hashes; manual annotations are separate. The currently running 2048 experiment
keeps its original 360 anchors and contributes no results to this audit yet.

For method design, separate discovering parameters from identifying behavior to
preserve. A useful selector must earn its value under the same protection policy
as random selection and LoRA. A subsequent coverage test should first compare
reviewed training-only anchors against exact-only anchors at the same replay
budget. The preserved targets should avoid unnecessary copied text. Decision-margin
protection is a separate intervention and must not be bundled into that first
comparison, since doing both would obscure which change helped.

Rebuild the automated queue with `python experiments/real_qa/audit_anchor_coverage.py`.
Its outcomes reflect whichever completed training-prediction files are available;
hashes and filenames identify the evidence. It is not a completion check for a live
experiment. Dataset text retains the SQuAD license described in ../DATA_LICENSE.md.
