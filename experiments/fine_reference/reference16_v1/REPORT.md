# FiNE works here, with limited transfer and measurable interference

Kaggle version 1 completed on 2026-10-20. The unmodified upstream editor gives a
clear selected-versus-uniform-random editing effect on the fixed first 16 KnowEdit
cases. This is a bounded reference comparison, not a full-paper reproduction or
evidence that NeuronPatch is better. Cases0–1 repeat the exposed smoke examples.
All 32 restoration checks pass, executed-source hashes match the submission, and
both smoke cases reproduce exactly in both conditions.

## Generated behavior

An answer-prefix diagnostic checks all supplied nested gold aliases after Unicode,
case, whitespace and leading-article/quote normalization, with a word boundary.
It does not judge the complete continuation. These are requested counterfactual
edits, not factual improvements.

| Probe category | Selected prefix matches | Random prefix matches |
|---|---:|---:|
| Original edit prompt | 15/16 | 0/16 |
| Reworded relation | 5/16 | 0/16 |
| Alternate subject names | 52/94 | 0/94 |
| Reasoning probes | 3/61 | 0/61 |
| Logical probes | 1/15 | 1/15 |

Counts pool probes for transparency; cases have unequal alias/reasoning coverage.
Case-weighted generation rates and complete raw outputs are saved in
[generation_audit.json](generation_audit.json) and results/. None of these prefix
counts establishes semantic accuracy of the whole answer. Several continuations
repeat themselves or invent additional claims.

Case 9 illustrates an answer-boundary issue: the requested name is “Johnny Reine,”
but the generated continuation starts “Johnny Reineke.” It gets 100% teacher-forced
target-token accuracy but fails the answer-word boundary. In contrast, one original
prompt passes the prefix diagnostic while receiving 83.3% token accuracy. The
diagnostics measure different things and must not be substituted for one another.

On rewordings, six selected outputs score at least 75% teacher-forced token accuracy
while missing the generated answer prefix. For example, the burial-place
rephrasing in case 10 scores 88.9% but begins “now a memorial garden” instead of the
requested place. The target prefix supplied during scoring makes later target
tokens easier; the generated first-answer branch remains wrong.
The original GPT-J metric also omits the EOS prepended for editing and generation.
Both conditioning differences matter; we have not attributed each disagreement
solely to teacher forcing. The paths are preserved for source fidelity, and
future candidate evaluation must also align token scores with generation inputs.

## Preservation, with a baseline-quality denominator

The 130 relation-specific probes contain only 18 baseline reference-answer prefix
matches. Selected edits retain 15/18 of those; random edits retain 17/18. These
are benchmark-reference answers, sometimes historical, not assertions of current
real-world facts. After editing, selected has 17/130 total prefix matches and
random 20/130; those totals include newly gained matches as well as retained ones.
The seven forgetfulness probes have zero baseline prefix matches, so conditional
answer retention there is undefined.

The three selected losses are concrete:

- Changing Janice Dickinson's sibling name also changes her citizenship answer
  from the benchmark's U.S. answer to Antigua and Barbuda.
- Changing Kanye West's mother's name to Geneviève Abelin makes the spouse prompt
  answer Geneviève Abelin as well, replacing the benchmark-listed spouse answer.
- Changing Prince Harry's sibling name to Shiban changes the religion answer from
  Church of England to Shinto.

The random control also loses a baseline match in another case: Kanye West's
mother answer becomes Kim Kardashian. Random selection is not intrinsically safe.
Exact continuation agreement is also not correctness: selected leaves 61/130
relation-specific continuations unchanged, random 38/130, yet most baseline
continuations do not match the reference answer.

These examples suggest relation spillover and target-token association as possible
failure modes. They do not establish a circuit explanation or prove that all
changes have one cause.

## Upstream teacher-forced metrics

Values average probes within category per case, then cases. Prediction agreement
is used for locality, as in the audited evaluator. This is not free-answer accuracy.

| Metric | Selected | Random | Cases / probes |
|---|---:|---:|---:|
| Rewrite token accuracy |98.96%|26.42%|16 /16|
| Rephrase token accuracy |73.62%|28.54%|16 /16|
| Alias token accuracy |82.99%|27.08%|15 /94|
| Reasoning token accuracy |41.94%|27.56%|9 /61|
| Logical token accuracy |21.11%|16.20%|9 /15|
| Relation-specific prediction agreement |86.12%|93.05%|16 /130|
| Forgetfulness prediction agreement |66.53%|96.43%|7 /7|

See [comparison.json](comparison.json) for full precision, per-case values, category
counts, and the pinned runtime/model/data manifest. One seed and a small exploratory
subset cannot establish a broad superiority or data-efficiency claim.

## Execution

- Source commit a2489cbac0dfe5a956385734065cfa041384e7d1; original editor/metric
  bodies unchanged. Source and data hashes are checked.
- GPT-J-6B, FP16 base/BF16 updates, CUDA0, two T4s allocated, one used by the model.
- Python 3.11, torch 2.0.1+cu117, Transformers 4.40.0. Both conditions use the same
  50-step cap, seed 42, k 5 per target token, coefficients and stopping threshold.
- Stage completion at 4350 seconds, about 72.5 minutes after kernel startup;
  model loading accounts for about 426 seconds. This is total wall time including
  downloads, scoring and generation, not a pure editing-time benchmark.
- [Integrity evidence](integrity.json) records 32 restore markers, terminal COMPLETE,
  exact submitted/executed bootstrap hashes, and identical smoke results.

## Decision

The old custom-QA failure did not refute FiNE: it changed the selector, model,
task, intervention axes, training objective and evaluation together. We now have
a working reference and a source-matched CPU library path. The full-size library
port check is running separately under its explicitly recorded comparison runtime.

The next quality work should address relation transfer and preservation of
already-working behavior. Add layer/multiplicity-matched random controls before
attributing the entire effect to exact neuron identity. A separate source audit
also found that applying a sparse mask after a protective projection can violate
that projection; jointly enforcing both constraints is an implementation candidate,
not a demonstrated model improvement. Use these exposed cases for development,
then freeze the candidate before fresh confirmation. Do not count better
teacher-forced scores alone as the requested breakthrough.
