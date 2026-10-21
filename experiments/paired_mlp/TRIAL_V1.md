# Bounded complete-answer comparison

**No pretrained outcome yet.** This tests paired incoming/outgoing updates
against outgoing-only, random and official LoRA at comparable parameter counts.
It is a development experiment, not a paper reproduction or fresh confirmation.
The [protocol](protocol_v1.json) and [inputs](inputs_v1.json) precede new inference.

## Task and scope

Reuse two exposed counterfactual edits: Leonardo DiCaprio citizenship to Syria,
and the Academy Award country association to Wassoulou Empire. These are
benchmark changes, not real-world truth or recommended deployments. The
empire/country type mismatch and noisy source references limit interpretation.
The prepared fresh eight-fact bank and new source indices remain unused.

Each edit has three complete training answers through EOS, one different
validation wording and four transfer wordings. Two transfer questions are
assistant-authored before inference. Eight transfer observations represent two
facts, not eight independent edits. Six full preservation answers train alongside
correction: three same-relation subjects and three utility questions.

Separate scope/utility validation banks choose rates. Four other subjects and
23 utility questions per edit test preservation; extra exposed scope probes
remain reported. The grid has1417 logical generation observations:109 base plus
1308 edited,77 distinct prompt IDs. Before fits, require at least18/23 complete
base utility-test answers; otherwise report an inadequate base and stop fitting.
This is a development capability check, not fresh confirmation.

Pre-inference data fixes expand Kamala to Kamala Harris and replace the ambiguous
Soviet Union training label with exposed Facebook/United States, removing that
probe from other banks. Accepted answers include country aliases, numeric word
forms and England for the club association. Scoring ignores case, whitespace,
enclosing quotes, trailing punctuation and leading "the". It never discards
an extra explanation. Non-stopping responses fail even with a correct start;
unexpected special tokens also fail. Generation is greedy, capped at32 tokens.

## Representation and training

Pinned Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, FP16 base, all36 MLP blocks.
This is the existing earlier QA checkpoint, not an unexplored stronger model.

| Family | Trainable scalars | Support |
|---|---:|---|
| Paired gradient |1,328,640|173 intermediate indices; gate/up rows and matching down columns|
| Paired random, seeds11/29/47 |1,328,640|Same counts within each layer|
| Outgoing gradient |1,328,640|519 down columns;3x paired count within each layer|
| Official LoRA |1,327,104|Rank1, alpha1, dropout0 on gate/up/down|

Allocate five paired indices to29 layers and four to seven layers, spread
deterministically. Rank the norm of the mean full-response gradient over the
three correction examples. Paired adds squared gate/up row and down-column
norms; outgoing ranks just the columns. All token positions contribute through
the graph. This is a group-gradient control, not new causal attribution.
Discovery cost is separate; matched scalar counts do not mean equal compute.

Every family/seed gets learning rates.0003/.001 and50 Adam updates with FP32
trainables. Each step uses one correction and one preserved complete response
in deterministic shuffled cycles: task CE plus preservation CE, both through
EOS. Fixed loss scale128, unscale then global norm clip1; no weight decay or
early stopping. Two independent edits give24 fits. Worker bound7200seconds,
outer worker-command timeout7500seconds; dependency setup time is additional.

## Selection and useful result

Rate selection sees validation banks only, jointly across both edits. Reject
rates that lose an initially reference-correct scope/utility validation answer
or newly insert the edited target there. Among eligible rates maximize complete
target-validation matches, minimize target-validation NLL, then prefer the
lower rate. Random seeds select independently. Report all rates, including
families with no qualifying candidate.

The paired candidate advances only with at least7/8 complete transfer answers,
at least two more than outgoing and the three-seed random mean, at least as many
as selected LoRA, and no losses of baseline-correct scope/utility test answers
or new target mentions there. All outputs must also be reviewed. If every
method reaches a ceiling, this provides no paired advantage. Do not change
scoring, rates or prompts after results. A count pass is not a quality recipe;
literal target scans do not prove semantic preservation.

## Verification

The actual24-fit worker runs on a tiny random Qwen3 in both FP32 and FP16 CPU
tests with the pinned dependencies. Compact reloads reproduce live teacher
logits exactly; merged models generate scored answers. Independent checks
decode every output, reconstruct complete-response losses, validate scalar
counts/support shapes and reject altered metrics. Original weight snapshots
and fingerprints restore the base after each fit, with two generation replays.
These are engineering checks, not evidence of language quality.

The real tokenizer verifies all109 input/answer rows and EOS without truncation;
maximum length57 tokens. No pretrained weights are downloaded on the laptop.
An in-place checked merge avoids allocating a second full model on the T4.

```bash
python -m experiments.paired_mlp.prepare --parent PARENT_INPUTS --output DATA
python -m experiments.paired_mlp.build --data DATA --output BUNDLE
# After downloading the private Kaggle outputs:
python -m experiments.paired_mlp.verify --results RESULTS \
  --protocol experiments/paired_mlp/protocol_v1.json --output INTEGRITY.json
python -m experiments.paired_mlp.analysis --results RESULTS --output SUMMARY.json
```

Inspect all results before the next hypothesis. Useful-library, data-efficiency
and paper-superiority claims still require fresh cases and an actual advantage.
