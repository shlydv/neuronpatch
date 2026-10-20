# Evidence-pair v1: baseline ceiling, no 4B adaptation fits

**The unchanged model already solves this controlled missing-evidence task too
well to evaluate improvement.** The frozen train/calibration-only headroom rule
stopped all twelve proposed adaptation fits. This is neither success nor failure
of the new objective or guided neuron selection: neither was trained at 4B here.

| Base model result | Training | Calibration |
|---|---:|---:|
| Missing evidence: exact UNKNOWN | 60/64 | 31/32 |
| Supported answer: strict normalized exact match | 60/64 | 28/32 |
| Retained-fact questions: strict exact match | not generated | 57/64 |

The selected calibration-only first-token UNKNOWN bias is **zero**. Adding a
refusal bias does not improve the constrained calibration choice. There were four
training missing-evidence errors and one calibration error; the protocol required
at least eight and four respectively. Validation and unrelated utility inputs
were not evaluated. Do not quote these training/calibration results as held-out
validation quality or as a HellaSwag result.

## What the actual answers say

The following is an **assistant audit after seeing predictions**, not independent
human gold. Frozen labels, aliases and primary scores are not changed.

All four calibration supported EM misses convey the correct answer: “the future
caliph Muawiyah I,” “In September 1695,” “In 1644,” and “Eirós Cave, in the
municipality of Triacastela.” All seven retained-fact EM misses also convey the
answer, with extra wording such as “specificity theory” versus “specificity.”
Thus the calibration positive-answer deficits are predominantly output-span
compliance, not the missing semantic capability we want to improve.

Training supported misses comprise three valid expanded answers and one UNKNOWN
where the supplied text implies the seller was Kirk Kerkorian, but identifies
the studio indirectly. The latter is a harder anaphora/context case rather than
a broad missing-evidence failure.

The five missing-evidence misses require separate interpretations:

| Case | Output | Audit |
|---|---|---|
| Russell group formation date, train | June 1879 | Wrong event/date; the formation date was removed. |
| Captain Britain debut date, train | Captain Britain debuted exclusively in the UK. | Related statement that supplies no date; fails the UNKNOWN instruction, without inventing a date. |
| Hungarian army leader at Mohács, train | Louis II of Hungary | The surviving account of his death in the battle makes this a plausible inference; label ambiguity remains despite source-only review. |
| Bell's father's recovery destination, train | The New World | Surviving family-migration evidence makes this plausible; the recovery premise is not explicitly supported. Another ambiguous case. |
| Swaziland House of Assembly election date, calibration | 9 September 1964 | Conflates a different legislative event with the requested later event. |

This gives a narrow actionable clue—confusing related events and their dates—
but too few clean cases to train and demonstrate reliable improvement. It does
not justify relabeling after the fact or constructing a test from just these
observed mistakes.

## What was executed and verified

Pinned Qwen/Qwen3-4B-Instruct-2507, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, FP16 on a standard free Colab T4.
Core dependencies are the frozen protocol's Torch 2.6.0 / Transformers 4.56.2 /
PEFT 0.17.1. Colab is the already-authorized fallback from the cancelled Kaggle
queue; no Kaggle result is claimed.

The source-only curation and complete protocol were frozen before predictions.
The actual CUDA preflight trained tiny Qwen models through selected-row, random-row
and LoRA paths, checked finite losses, exported/merged updates and restored the
base exactly. This is a software test, not a 4B quality experiment.

The 4B baseline/calibration phase completed normally. The final full base-weight
fingerprint matched the original:
`3b369c58239c915efbf22934957e32f5134a697729ccad78f2d8e7b29b76ee5c`.
The full outputs were downloaded and their source/data hashes verified before
releasing the temporary GPU runtime. No worker remains active.

See [integrity.json](integrity.json), [completion.json](completion.json),
[headroom.json](headroom.json), [calibration.json](calibration.json) and
[actual answers](base_predictions.json). The public submission contains the
executed source/data hashes; raw passages and binaries remain ignored artifacts.

## Consequence for the library goal

The mathematical pairwise term, matched-budget runner and reviewed data are usable
experimental infrastructure. They are not a demonstrated improvement over FiNE,
NeST, SafeNexus, LoRA or a simple output bias.

The earlier verified FiNE reproduction already establishes that selected neurons
can work on its knowledge-editing setup. This new result instead says our latest
controlled abstention setup has insufficient headroom. Making this dataset
smaller or tuning more rates will not resolve that.

Before the next training comparison, establish a sufficiently common **semantic**
failure on a fixed external diagnostic. One suitable next candidate is
[FaithEval](https://github.com/SalesforceAIResearch/FaithEval), which supplies
unanswerable, inconsistent and counterfactual contexts. Use a separate development
slice for feasibility, keep another slice untouched for confirmation, and inspect
whether failures involve evidence use rather than answer wording. Do not assume
it is difficult for this exact model, treat its test split as training data, or
claim the current UNKNOWN-specific objective already handles its other tasks.
