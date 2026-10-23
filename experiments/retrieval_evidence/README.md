# Qualify an evidence-use task before more adaptation

This is a development qualification experiment, not a new trained method.
The unchanged-model baseline was submitted to Kaggle on 23 October 2026. The previous two-fact diagnostic is closed; see the
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

## Completed qualification

An output-independent SHA256 ordering selected 32 candidate questions from all
300 source records. The private packet contains their full question, answer
groups, unchanged documents and provenance. The [completed review](review_v1.json)
retains 22 and excludes 10, without replacements. Every complete document set
was read before model inference. This is assistant review, not independent human
adjudication. Exclusions include surviving explicit/implicit answers and
ambiguous dates or release types.
No model output was used for selection. The baseline completed with worker exit 0 as version 1 of
[`shlydv/neuronpatch-retrieval-evidence-v1-20261023`](https://www.kaggle.com/code/shlydv/neuronpatch-retrieval-evidence-v1-20261023).
The downloaded execution source matches the [frozen SHA256](frozen_v1.json).

The reproducible [input builder](inputs.py) produces 110 queries: ordinary and
explicit evidence instructions on each supported/missing pair, plus one
question-only control per question. [Token checks](input_checks_v1.json) use the
pinned tokenizer offline: longest input 1,115 tokens, no truncation, no local
model weights. Archived-document instructions prevent treating historical CEO
or price answers as claims about today. Do not rewrite these questions after
seeing model failures.

The frozen baseline-only comparison uses the existing pinned 4B
generation infrastructure: supported documents, missing documents, and a
question-only factual control. The last control measures whether ignoring the
documents already answers the questions; it has a different instruction contract
and is not itself an abstention test. A stronger evidence-only prompt must be
compared before attributing headroom to a missing weight update. Full answer
correctness and supported/missing separation are distinct measurements.

The [completed baseline](complete_v1/REPORT.md) does not meet its frozen
headroom criterion: 21/22 correct abstentions and only one repeated event-relation
error. No adaptation follows on this slice.

Training is warranted only if reviewed outputs expose a repeated, meaningful
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
