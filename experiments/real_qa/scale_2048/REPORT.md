# More data helps the measured trade-off, but does not solve preservation

Both prespecified one-epoch fits completed successfully on 2026-10-19 (process
exit 0). The 128/512/2048 curve is complete. All fits use the same model and
256 development questions, with supported and impossible counts of 128 each.
Training steps and replay coverage increase with dataset size; this is not an
equal-compute data-diversity experiment.

| Training size / learning rate | Supported EM /128 | Unnecessary refusals /128 | Correct missing refusals /128 | Base-correct retained /45 |
|---|---:|---:|---:|---:|
| Base | 45 | 13 | 34 | 45 |
| 128 / .0001 | 40 | 47 | 81 | 34 |
| 128 / .0003 | 43 | 33 | 62 | 36 |
| 512 / .0001 | 31 | 75 | 107 | 26 |
| 512 / .0003 | 46 | 56 | 95 | 37 |
| 2048 / .0001 | 50 | 43 | 94 | 38 |
| 2048 / .0003 | 50 | 48 | 96 | 33 |

The larger sets can improve conditional behavior rather than only increasing
refusal, but the effect depends on learning rate and size. The final two fits
share the same supported EM yet preserve different baseline answers. The higher
rate trades five additional lost base-exact answers for two additional correct
missing-evidence refusals. It also changes a correct chicken answer into the
contrasted therian mammal and replaces a correct explanation with a nonanswer.

On training, the lower/higher rates reach 438/513 supported matches and 883/937
correct missing refusals out of 1024 each. They preserve 300/303 of 360 exact
anchors; 43/37 anchors become refusals. Preservation therefore still fails on
explicitly protected training examples. Expanding coverage cannot be assumed to
solve all interference.

## Read the answers, not only the aggregate

Both semantic_review files inspect every supported EM change relative to base,
with full contexts and before/after outputs. Reviews are assistant qualitative
judgments, not blinded semantic evaluation; no automatic scores are changed.

The lower-rate net gain of five is 12 new matches minus seven losses. Gains include
three clear corrections, three precision changes, three wording changes, two
ambiguous alternatives and a question/reference mismatch. Six formerly exact
answers become refusals; one remains valid with an extra adjective.

The higher-rate net gain of five is 17 new matches minus 12 losses. Gains include
five clear corrections, four precision changes, three wording changes, three
ambiguous alternatives and two question/reference mismatches. Losses include
eight clear answer-to-refusal changes, one refusal on a question with a questionable
presupposition, two substantive wrong/nonanswers and one valid wording change.
These selected changes do not estimate overall semantic accuracy. Unchanged EM
can hide additional semantic gains or losses.

Article-bootstrap development intervals for supported EM differences include zero
at both rates. The added correct refusals and unnecessary refusals are much clearer
than the small supported-score increase. See ../scaled/uncertainty.json and its
protocol for sampling assumptions and limitations. These are exploratory results,
not evidence of superiority across seeds, models or task families.

## Verification and next action

Executed source archives match the recorded source hashes; dataset hashes and
prediction populations match. Recomputed metrics match the summaries, every fit
contains 1024 finite loss records, and both adapters contain 829,440 finite
parameters. integrity_checks.json records these checks. These checkpoints have
not undergone a separate reload-generation comparison; do not claim that check.
The original active-run source was archived before later runner changes.

The next controlled test adds 24 reviewed valid training answers to the 512-prefix
preservation pool while holding replay compute fixed. It is intended to isolate
coverage from update geometry; it does not introduce a new neuron selector or
cluster sharing. Confirmation is still unused. The full method/library goal
remains open: neither the learning curve nor numerical component tests establish
a superior combined method.
