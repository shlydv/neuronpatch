# External faithfulness probe: interference is real; adaptation is not yet justified

**Executed unchanged-model development diagnostic, not a new editing result.**
Pinned Qwen3-4B gets 43/48 original ARC reference answers without the passage.
With fabricated passages and instructions to use established science, it loses
nine of those correct answers, with one further answer unresolved. Moving the
identical instruction into the system role still loses 11/43. However, ignoring
the passage is already an effective no-training baseline here. No weight edit
has been trained or shown to beat it.

The frozen strict parser substantially understates content accuracy. Source
review also finds questionable reporting and missing-evidence labels. These
problems must not become evidence that neuron selection fails or a new method
works. This probe does not establish a clean adaptation target.

## Execution and integrity

- Google Colab free Tesla T4, **not Kaggle**; authorized fallback after the
  cancelled queue. Notebook and results saved before runtime release.
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, FP16, eager attention,
  Torch 2.6.0+cu124, Transformers 4.56.2, greedy decoding.
- Initial 300 frozen queries: 481.72 seconds of worker time, zero training steps.
- Every one of the 14 token-limit-truncated responses was rerun at 128 tokens;
  all 12 still-truncated responses were then rerun at 1024. Worker times:
  206.87 and 345.86 seconds. All preceding generated token sequences reproduce
  exactly. One answer remains truncated at 1024 and is unresolved.
- Exploratory fixed system-role control: all 48 original questions in both
  explicit-report and science modes, 96 queries, 317.07 seconds, max 1024 new
  tokens, no truncations. Instruction words, document, question and choice order
  preserved. This is one role control, not wording optimization or confirmation.
- Full initial/final base fingerprint in both main runs:
  `3b369c58239c915efbf22934957e32f5134a697729ccad78f2d8e7b29b76ee5c`.
  Extension workers verify that same base and unchanged final weights.
- Local verification checks source/data hashes, byte-exact role-input
  reconstruction, every generated prefix, counts and downloaded archives. See
  [integrity.json](integrity.json), [identity](primary/identity.json),
  [completion](primary/completion.json) and individual execution logs.

The initial 32-token output cap was insufficient despite the one-letter request.
The two uniformly applied extensions are a separately labeled evaluation repair.
They do not replace frozen outputs or strict metrics. Counting just the initial
letter or first 32 tokens misses self-corrections, including a correct final
answer after 990 tokens.

## Primary strict results, unchanged

| Task / frame | Strict correct | Unparsed | Truncated at 32 |
|---|---:|---:|---:|
| Reporting, ordinary | 2/48 | 46 | 1 |
| Reporting, explicit | 9/48 | 32 | 0 |
| Established science with untrusted passage | 2/48 | 44 | 12 |
| Science without passage | 31/48 | 16 | 0 |
| Reporting ordinary, rotated choices | 1/24 | 23 | 1 |
| Reporting explicit, rotated choices | 9/24 | 14 | 0 |
| Missing evidence | 16/30 | 14 | 0 |
| Conflicting evidence | 11/30 | 16 | 0 |

These measure literal response format, **not semantic accuracy**. For example,
a correct `D. [full option]` fails the one-letter parser. They cannot by
themselves establish the semantic-error feasibility threshold. See
[primary outputs](primary/predictions.json) and
[strict_summary.json](strict_summary.json).

## Secondary option-content diagnostic

The secondary rule accepts a bare letter, exact option text, or matching letter
plus exact option text. It uses no gold-dependent extraction. Explanations,
label/text disagreements and self-corrections require explicit assistant review,
recorded in [content_reviews.json](content_reviews.json). This is post-run
assistant adjudication, not independent human gold.

| Task / frame | Reference/target content matches | Unresolved |
|---|---:|---:|
| Science without passage | 43/48 | 0 |
| Science with untrusted passage, user role | 35/48 | 1 |
| Same science instruction, system role | 33/48 | 0 |
| Reporting, ordinary | 32/48 | 0 |
| Reporting, explicit | 33/48 | 0 |
| Same explicit reporting instruction, system role | 30/48 | 0 |
| Reporting ordinary, rotated choices | 18/24 | 0 |
| Reporting explicit, rotated choices | 19/24 | 0 |

Reporting rows are matches to published fabricated targets, **not fully validated
semantic faithfulness**. Two inspected examples expose source-scope problems:
a flight-distance question paired with structural-maintenance millimeters
(`Mercury_7217368`), and an evaporation passage that supports unchanged molecule
count while the target asserts unchanged intermolecular distance
(`MDSA_2007_8_24`). Frozen targets were not changed after predictions.

The unresolved science answer puts the user-role reference count at 35–36/48.
All four frames are paired views of the same 48 source questions, not independent
samples. Rotation is only a 24-case diagnostic, not proof of order robustness.

Among the 43 questions reference-correct without the passage:

| Science with passage | Still correct | Changed to wrong | Unresolved |
|---|---:|---:|---:|
| User instruction | 33 | 9 | 1 |
| Same instruction in system role | 32 | 11 | 0 |

A clear case is `VASoL_2008_3_25`: without the passage, touch and sight for
measuring hair; with the fabricated passage, smell and hearing despite the
science instruction. Some other ARC alternatives are debatable, so the aggregate
is reference-answer interference, not proof every changed answer is impossible.

The comparison changes instruction and passage presence together; it does not
isolate a neuron or establish passage tokens as the sole cause. The fixed
system-role control rules out that particular simple repair here.

For the explicit factual mode, routing to the already evaluated no-context
template scores 43/48. Dropping a passage also removes facts available only in
that passage. A useful adaptation test must include necessary-context cases and
both reporting and factual modes, or show a benefit beyond this simple route.

## Missing and conflicting evidence audits

All 14 strict missing-evidence mismatches were checked against full contexts.
The [audit](unanswerable_audit.json) identifies three clear unsupported assertions:
first-mentioned versus first touchdown, considerable versus most attention, and
a vessel's current versus hijacking location. Other mismatches: three nonanswers,
three underspecified role answers, two supported alternatives/bad labels, one
ambiguous fragment, one valid expanded abstention and one malformed clue. These
categories sum to 14. The narrower clear-unsupported-assertion count does not
establish the six-error feasibility criterion; other errors are kept separate.

The 19 strict conflicting-evidence mismatches have not all received complete
source-level adjudication. At least one is a valid expanded CONFLICT response;
some questions allow multiple answers rather than a logical contradiction.
The 11/30 strict score is not evidence for 19 genuine conflict-detection failures.

## Decision and connection to the library goal

No adaptation was launched on these exposed benchmark inputs. Audited counts
do not establish the reporting feasibility threshold. Science-mode interference
is exploratory, not a post-hoc replacement for that threshold. Unselected
benchmark IDs and previous evidence-pair validation remain unexposed by this probe.

Concrete implementation lessons: retain complete token sequences, verify
evaluation extensions, separate format from content, inspect source scope, and
evaluate no-training controls before claiming a learned improvement. Tests now
enforce extension and parsing invariants. This improves the experimental
foundation; it is not the requested superior adaptation method.

Do not run another UNKNOWN-training sweep or tune prompts on this slice. A
future context-use method needs a checked task requiring useful supplied evidence
while rejecting irrelevant or contradictory evidence, with no-context and
instruction/routing baselines. Training material must be separate. The parallel
[FiNE update audit](../../fine_reference/geometry_v1/REPORT.md) provides a measured
compactness hypothesis, without a behavioral compression result yet.

## Reproduce analysis without a GPU

Install experiment dependencies and `pyarrow==19.0.1`. Download the exact
counterfactual parquet at the revision/hash in
[data_manifest_v1.json](../data_manifest_v1.json). Raw passages are not
redistributed here.

```sh
python -m experiments.faithfulness_probe.analyze \
  --results experiments/faithfulness_probe/colab_v1/primary \
  --output /tmp/faithfulness-strict
python -m experiments.faithfulness_probe.content_report \
  --source /path/to/pinned-counterfactual.parquet \
  --output /tmp/faithfulness-content
```

The content command reads committed outputs by default, verifies source bytes
and both extension chains, and reproduces public counts. Every content decision
records exact response SHA256 and primary/extension/system-role provenance.
Original strict predictions stay untouched. Reviews, source pins, generation
tokens, completion records and download hashes are public.

This is a length-limited development screen, not an official full FaithEval score.
Dataset attribution and license cautions are in the [README](../README.md).
