# Qualify an evidence-use task before more adaptation

This is preparation for a development experiment, not a new trained method or
an announced GPU run. The previous two-fact diagnostic is closed; see the
[completed result](../alias_supervision/complete_v1/REPORT.md).

## Product question

Can a compact patch improve answering from retrieved documents when similar
but nonanswering documents are present, while retaining correct supported answers
and unrelated capabilities? The desirable change is conditional evidence use,
not simply more abstention or learning the benchmark's factual answers.

The original SQuAD experiments already covered natural unanswerable questions.
The later sentence-removal pairs had little base-model headroom. Repeating either
without acknowledging those results would not answer a new question.

## Candidate source and inspected limitations

[RGB](https://github.com/chen700564/RGB) provides retrieved positive and negative
documents for the same questions. Its refined English source corrects some
earlier document and answer annotations. We pinned revision
`65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615` and downloaded its source for inspection;
we did not execute or incorporate its evaluator.

Inspection establishes two limitations to handle explicitly:

- Its answer matcher uses substrings, which cannot distinguish an asserted answer
  from a negated or contradicted mention. Full-output assessment is required.
- Source question 228 asks who the Twitter CEO is without a date; the positive
  document announces Yaccarino, while a negative document explicitly describes
  Musk becoming CEO. Negative annotation alone does not establish that the
  supplied documents cannot answer an underspecified question. This example is
  an audit warning, not a model failure or a claim about the current CEO.

Source answers have **AND across outer items, OR across nested aliases**.
Flattening them would wrongly accept a partial answer for multi-part questions.
The independent preparation code preserves that distinction. It also retains
whole documents, balances their count, and replaces exactly one slot between
supported and missing conditions. It does not match document lengths or certify
absence of an answer. Those limitations must remain visible.

The upstream README licenses code and data CC BY-NC-SA 4.0. Downloaded documents
and derived review packets remain private research artifacts under that source
license; they are not added to the library's permissively licensed code or offered
as commercially usable training data. Our preparation module is independently
written. Commercially reusable training data would be a separate product requirement.

## Current state and next concrete step

An output-independent SHA256 ordering selected 32 candidate questions from all
300 source records. The private packet contains their full question, answer
groups, unchanged documents and provenance. All are explicitly unreviewed.
No model output has been used for selection and no new GPU run has been launched.

Before inference, inspect each candidate for explicit positive support, alternate
answers in negative documents, time ambiguity, conflicting snippets and token
budget. Record every exclusion rather than quietly replacing difficult examples.
Do not rewrite source questions after seeing model failures.

Then freeze a small baseline-only comparison using the existing pinned 4B
generation infrastructure: supported documents, missing documents, and a
question-only factual control. The last control measures whether ignoring the
documents already answers the questions; it has a different instruction contract
and is not itself an abstention test. A stronger evidence-only prompt must be
compared before attributing headroom to a missing weight update. Full answer
correctness and supported/missing separation are distinct measurements.

Training is warranted only if the reviewed outputs expose a repeated, meaningful
failure after simple prompting. These development examples cannot then serve as
fresh confirmation. No automatic training follows from a low substring score.

## Method hypothesis, not another unrelated benchmark

The paper-inspired question is whether selecting components for **conditional
evidence discrimination across examples**, rather than raw answer/refusal strength,
creates a more useful sparse update. The existing evidence-pair ranking objective
offers a limited way to exclude a global refusal bias; it is not a new algorithm
and does not guarantee complete-answer correctness. FiNE-style localization,
NeST-style sharing and preservation-aware updates remain ingredients to test,
not benefits to assume or combine blindly.

First establish the task and an ordinary LoRA learning baseline. Any sparse
candidate must use the same supervision and tuning opportunities as random and
LoRA controls. Compare supported answers, appropriate abstention, unrelated utility,
discovery cost and patch size. If LoRA delivers the same improvement, attribute it
to the objective/data rather than neuron selection. No claim of novelty, superiority
or general behavioral robustness follows from preparing this packet.
