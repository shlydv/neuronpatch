# A repeated relation-specific failure, with intact extraction controls

The frozen development qualification passes. The unchanged model gets **75/96**
full responses correct, including **24/24 target-only controls**. Four of twelve
family/name cells have supported-answer errors, spanning interview and release
families; the predeclared minimum was three cells across two families. These
are four templates with correlated variants, not 96 independent tasks.

| Family | Target last | Target first | Target absent | Target only |
|---|---:|---:|---:|---:|
| Tournament: reach vs play semifinal | 6/6 | 6/6 | 6/6 | 6/6 |
| Interview: qualify vs final assessment | 4/6 | 0/6 | 0/6 | 6/6 |
| Grant: shortlist vs winner | 6/6 | 6/6 | 6/6 | 6/6 |
| App: announce vs release | 5/6 | 6/6 | 0/6 | 6/6 |
| Total | 21/24 | 18/24 | 12/24 | 24/24 |

## What actually goes wrong

The interview documents say that one person assessed an applicant **to qualify
for** the final interview and another assessed them **during** the final interview.
When the final-interview passage comes first, the model refuses all six cases,
usually explaining that the records contradict each other. They describe two
stages, so this is a false conflict. Reversing passage order repairs four of six.
Yet without the final-interview passage, it names the qualifying assessor in all
six cases instead of returning UNKNOWN.

The release cases show a different asymmetry. The model generally uses an
explicit release attribution correctly, but without it always substitutes the
person who announced the app. The evidence does not establish who released it.
One supported release case returns UNKNOWN with the target last; its reversed
order and role-swapped controls succeed.

The tournament and grant families are solved, including both name assignments
and passage orders. In particular, short fictional tournament examples do not
reproduce the original long-document tennis error. This does not establish
whether factual associations, context richness, length or their interactions
cause that original failure.

These contrasts make a global abstention shift a poor solution: the model needs
to answer more on false-conflict supported cases and abstain more on missing-target
cases. Correct target-only answers show the task is not simply inability to read
the requested names. No circuit or causal neuron set has been identified.

## Review and verification

All [96 saved responses](predictions.json) were read against the
[public synthetic inputs](../inputs_v1.json). Full sentences and possessives that
correctly identify the answer receive credit. The [review ledger](review.json)
records each decision and text hash. This is assistant review, not an independent
human annotation or blinded scientific evaluation.

The [independent checker](integrity.json) verifies frozen prompts/protocol,
all decoded text, first-token argmax agreement and unchanged weight fingerprints.
Every response reaches EOS; none is cut off. The
[Kaggle version 1 job](https://www.kaggle.com/code/shlydv/neuronpatch-relation-contrast-v1-20261023)
completed with exit 0, 119 seconds of worker computation, and zero training.
The accepted source matches the frozen uploaded hash. Full logits remain local;
public tokens/text and the pinned tokenizer permit response review and decoding.

## What this warrants next

This establishes a small, labeled development weakness suitable for a controlled
learning pilot. Freeze new name sets and independent phrasings before fitting.
Keep correction, missing-evidence abstention and the already-solved control
families separate; test both passage orders. Train an ordinary LoRA baseline
with the same complete-answer labels before attributing improvement to neuron
selection. A more-refusal-only change must fail the supported-answer checks.

The current 96 cases are exposed diagnostics and cannot become the final fresh
confirmation set. A successful next pilot would still require transfer to new
relations and natural documents, unrelated utility, multiple seeds and total-cost
comparisons. This qualification is not an adaptation result, a published-paper
replication or a selected-neuron advantage. It is a concrete problem definition
that avoids interpreting arbitrary answer changes as progress.
