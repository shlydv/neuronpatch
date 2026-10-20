# Upstream FiNE editing works on the two-case smoke test

Completed on Kaggle, version4, inspected 2026-10-20. This establishes functioning
editing and an observed selection advantage on two fixed examples; it does not
reproduce a complete paper table or establish a new method.

## Actual generated answers

| Prompt category | Selected neurons start with requested target | Random neurons |
|---|---:|---:|
| Original edit prompt | 2/2 | 0/2 |
| Alternate entity names | 10/10 | 0/10 |
| Rephrased relation | 0/2 | 0/2 |

Targets are deliberately counterfactual: changing DiCaprio's citizenship to Syria
and associating Best Picture with the Wassoulou Empire. These are requested model
edits, not claims that those statements are true. Prefix success also does not
certify the rest of the generated text. Continuations contain invented details
and repetition. Both rephrasing generations miss the target, correcting the
initial informal description that only one failed.

The second rephrasing nevertheless scores **75% teacher-forced token accuracy**:
its generated answer never starts with Wassoulou Empire. This is a concrete
example of why teacher-forced metrics and generated behavior must both be shown.

## Baseline quality limits the preservation conclusion

A subsequent [free-generation audit](generation_audit.json) checks all supplied
gold aliases, with case/whitespace/leading-article normalization and answer-word
boundaries. It reproduces the rewrite, alias and rephrase counts above. Across
the 10 relation-specific probes and one forgetfulness probe, the **unedited model
has zero answer-prefix matches**. Inspection also finds wrong or non-answering
continuations, such as calling DiCaprio's birthplace a mystery while inventing an
Italian birthplace, or saying his citizenship is unknown. One relation probe is
itself awkwardly worded: an award being asked which award it won.

Consequently, this sample cannot measure retention of already-correct generated
answers: that denominator is zero. Matching old teacher-forced predictions or
unchanged continuations is still a valid stability diagnostic, but does not show
that useful knowledge was preserved. The prefix heuristic is not a whole-answer
judge; inspect its saved raw examples. A useful library claim will require a
healthy base task and enough correct baseline answers, alongside edit success.

## Official scoring functions, same paired inputs

| Metric | Selected | Random |
|---|---:|---:|
| Rewrite token accuracy | 100% | 12.5% |
| Alias token accuracy | 100% | 12.5% |
| Rephrase token accuracy | 37.5% | 12.5% |
| Reasoning token accuracy | 22.9% | 7.4% |
| Logical token accuracy | 0% | 0% |
| Relation-specific prediction agreement | 85.4% | 100% |
| Forgetfulness prediction agreement | 50% | 100% |

Values average probes within category and then cases; only one case supplies
forgetfulness/logical probes. The random condition preserves predictions while
failing to install the requested edits. Agreement does not establish correctness
or preservation of an entire generated continuation. On this tiny sample the
correction/preservation trade-off qualitatively resembles the paper's ablation.

## Execution and integrity

- Official editor and score function bodies are unchanged, pinned at source
  commit `a2489cbac0dfe5a956385734065cfa041384e7d1`.
- GPT-J-6B and KnowEdit revisions/data hash are recorded in `results/manifest.json`.
- Python3.11, torch2.0.1+cu117, transformers4.40.0, FP16 model, upstream BF16 deltas.
  NumPy/Accelerate compatibility versions differ from the complete upstream pins.
- All modules on CUDA0 avoids the upstream hook's cross-device multiplication
  failure. The successful job allocated two T4s but used CUDA0 for the model.
- Both conditions use the same 50-step cap, learning rate, losses and per-token
  neuron budget. Upstream resets seed42; no multi-seed result is established.
- Every restore check passes and execution reaches `STAGE complete`; kernel
  status is COMPLETE. Runtime to completion is about 930 seconds, including about
  455 seconds in model loading. It is not a pure editing-time benchmark.
- Pulled version4 source embeds exactly the locally audited harness files.
  Paired requests, probes, baselines and baseline scores match. File checksums
  are recorded in `integrity.json`; raw outputs and execution log are retained.

## What was missing from our earlier comparison

Our earlier work changed several things together: a 0.5B instruction model,
a document-grounding/refusal task, global selectors, restricted layer coverage,
parameter axes and a different preservation objective. FiNE instead selects
per-edit/per-target-token supports across eligible layers and edits outgoing
weights using its own composite loss. This smoke test does not isolate which
change explains the difference; it shows the published recipe can work in our
execution environment. Earlier negative custom-QA results did not refute it.

## Next decision

The frozen first16-case comparison is running with the same recipe. The original
two cases are included, so this is expanded exploratory evidence rather than an
independent confirmation set. Inspect correction, aliases, full rephrasings,
reasoning and actual interference together. If the pattern holds, integrate a
faithful editor into NeuronPatch before modifying its selection or training.
Candidate improvements should address demonstrated weak transfer/preservation
and compete against the unmodified editor and equal-budget random controls.
