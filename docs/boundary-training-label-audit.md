# Reviewed labels before the next baseline

The earlier instruction-boundary experiments deliberately distilled every completed
clean-model response, including its factual mistakes. Their reports acknowledged
teacher-error retention, but that recipe is unsuitable for using an answer as an
unqualified preferred factual target in the next experiment. Replacing teacher
outputs with the original dataset reference is also insufficient.

We inspected all32 training and16 calibration tables before any new training.
This is an assistant source review, not independent human annotation. Original
confirmation judgments, protocols and historical results remain unchanged.

Concrete issues include:

- A teacher names the final album track instead of the final track credited to
  the requested producer. Another counts zero songs below two minutes although
  two qualifying lengths are visible.
- A home-win difference has a false premise and an incorrect reference. The
  table contains four wins in the earlier season and one in the later season;
  the clarified question asks how many fewer, with answer three.
- An artists-count question confuses qualifying rows with distinct artists.
  Four qualifying rows represent three artists; the teacher gives two.
- Two questions need flags or nationalities absent from the textual tables.
  Those documents are excluded from this new training/calibration bank.
- “Before,” “next to,” missing numeric entries, consecutive wins, walkovers and
  inclusive year counts can make labels ambiguous. The revised questions specify
  the intended source operation instead of silently choosing an interpretation.

The review keeps31 training and15 calibration documents, clarifies eight training
and three calibration questions, and supplies concise source-supported targets.
All proposed arms must use these same reviewed labels. Changes from old runs then
include label quality and response style, so old adapters are not causal controls
for the new objective comparison. Teacher mistakes are not proved to cause the
previous attack-transfer collapse; the dominant distinction remains training
coverage and objective. Fixing labels avoids adding a second preventable problem.

## Competing real tasks

`experiments/instruction_boundary/reviewed_tasks.py` binds reviews to the entire
source case, preserving the table while making question revisions explicit. It
constructs balanced donor assignments within each split: every task is used once
as recipient and once as donor per round, with no self-pairs, equal-answer pairs
or repeated edges. Input ordering does not change the assignment.

The competing request asks a different actual table question, with its own table.
It moves across the outer quotation boundary; the two response alternatives then
swap preference. Thus both answers occur equally often as chosen and rejected.
The request text itself is unchanged across the two trust locations. These are
real lookup/count/comparison/arithmetic operations, but still one table-QA domain;
this is not diverse open-domain instruction training or a SecAlign reproduction.

Private preparation contains:

| Split | Reviewed documents | Marker preference pairs | Cross-task pairs | Combined pairs |
|---|---:|---:|---:|---:|
| Training |31|62|124|186|
| Calibration |15|30|60|90|

The original marker requests are retained in the combined bank. Every pair has
chosen and rejected complete responses. These are alternative data-coverage banks,
not186 new independently labelled documents. Split membership is disjoint.
Qwen's actual pinned tokenizer verifies all responses without truncation; maximum
prompt-plus-response lengths are1559 training and1610 calibration tokens.
Longer contexts increase cost and must be measured, not hidden under equal steps.

[Preparation receipts](../experiments/instruction_boundary/reviewed_task_preparation.json)
record hashes, counts and lengths. The [source-review record](../experiments/instruction_boundary/training_source_review_v1.json)
records decisions and source hashes without publishing the corpus or private
training targets. Both raw reviewed banks and the exact preparation scripts are
retained under ignored local artifacts for the next bounded experiment.

Eight focused tests pass across preference and reviewed-task modules, including
source-tampering rejection, explicit question revisions, balanced donor matching,
answer reversal, nested quotation parsing and actual tiny-model adapter gradients.
These checks verify construction and math, not learned performance. The integrated
training runner, frozen reference-score cache, new confirmation bank, launch
protocol and runtime/export checks remain to be completed before a GPU submission.
