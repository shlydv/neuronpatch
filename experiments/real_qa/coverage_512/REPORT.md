# Expanded preservation coverage is insufficient under the original decoder

Both 512-example fits completed with process exit 0. The only training treatment
is expanding the replay pool from 91 exact-correct examples to 115, adding 24
reviewed valid nonexact answers. Model, data, supervised order, two learning rates,
256 steps, loss and 512 total replay exposures match the controls. Added coverage
reduces replay exposure per individual example. This is not a pure increase in
protection strength.

| Rate / policy | Supported EM /128 | Unnecessary refusals /128 | Correct missing refusals /128 | Base-correct retained /45 |
|---|---:|---:|---:|---:|
| .0001 exact anchors | 31 | 75 | 107 | 26 |
| .0001 expanded | 30 | 73 | 103 | 25 |
| .0003 exact anchors | 46 | 56 | 95 | 37 |
| .0003 expanded | 39 | 63 | 100 | 35 |

No clear development trade-off improvement emerges. At the stronger rate,
increased correct refusals accompany worse supported-answer performance. Paired
article-bootstrap intervals are in comparison.json; they describe development
sampling, not independent confirmation or seed variation.

Review of all 24 added training answers finds valid answers change from 8 to 8
at the lower rate and 12 to 13 at the higher rate. The lower-rate treatment recovers
two refusals but loses a correct answer to refusal and another to a truncated
related response. Exact-match changes also include harmless wording changes.
Original exact-anchor retention changes 53→56/91 at the lower rate, but 86→80/91
at the higher rate. A wider protection pool alone has not solved the problem.
All semantic annotations are assistant judgments, not blinded evaluations.

## Decoding audit changes the interpretation

The raw first-token diagnostic on all 115 anchors found saved refusals whose raw
argmax was still a plausible answer token. The model's inherited generation
configuration contains repetition_penalty=1.1. Greedy decoding disables sampling,
but does not neutralize this logits processor. The training preservation loss
operates on raw logits; reported generated answers use processed logits.

A bounded same-checkpoint audit of five discrepancies and six nonrefusal controls
reproduces all 11 original outputs. Setting only repetition_penalty=1.0 recovers
valid answers in all five selected discrepancies. Neutral processed logits match
the raw forward logits within 1e-4. One control becomes wordier, adding the year
before the requested paper title; the other controls retain their answers.
This establishes a decoder contribution on selected examples, not a general fix.
The audit is intentionally selected and cannot estimate population benefit.

This was a missing evaluation control in our workflow. Do not attribute every
refusal directly to neuron selection or the training loss. Conversely, raw logits
also change substantially on some anchors, so the decoder is not established as
the sole source of damage. The next read-only evaluation applies neutral repetition
penalty to the base and all four saved checkpoints on the same 256 development
questions. It will preserve original scores and check both answers and refusals
before deciding whether the evaluation policy or objective should change.

The first-token diagnostic does not measure whole-response KL or semantic quality.
No new loss, selector or shared update was tested here. Confirmation remains unused;
the broader method/library goal is still open.
