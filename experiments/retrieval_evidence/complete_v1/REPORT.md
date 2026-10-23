# Retrieved-evidence qualification: insufficient training headroom

**Decision: do not train on this slice.** The frozen qualification required
at least four of 22 questions with a clear evidence-use error after stronger
instructions, while retaining at least 18 supported correct answers. Only one
question shows that repeated error. The planned adaptation comparison is not
warranted by this result; no training was performed.

The unchanged Qwen3-4B-Instruct-2507 completed all 110 queries on Kaggle version 1
of [this run](https://www.kaggle.com/code/shlydv/neuronpatch-retrieval-evidence-v1-20261023).
Worker computation took 231 seconds, provider bootstrap plus worker 337 seconds.
The [independent artifact check](integrity.json) verifies exact frozen inputs and
protocol, all decoded responses, finite first-token vectors, all 110 first-token
argmaxes and unchanged before/after model fingerprints. These checks do not prove
model quality. [Review](review.json) covers every complete saved response.

| Measure, 22 questions per condition | Ordinary instructions | Explicit evidence instructions |
|---|---:|---:|
| Missing evidence: correct UNKNOWN | 21 | 21 |
| Supported: correct content and completed answer at reference specificity | 20 | 19 |
| Supported: correct content but cut off mid-sentence | 1 | 1 |
| Supported: supported city, less specific than reference venue | 0 | 1 |
| Supported: wrong event round | 1 | 1 |
| Both supported and missing correct, complete, at reference specificity | 20 | 19 |

The explicit supported answer “Los Angeles” is a defensible answer to a broad
“where” question, though the reference names the hotel. Counting it as acceptable
raises explicit complete supported/paired success to 20/22. It is not a clear
model error or evidence for a selected-neuron training advantage.

## The actual failure

Source question 136 asks whom Alcaraz beat **in** the 2022 US Open semifinal.
Both instruction variants answer Jannik Sinner, in both evidence conditions.
The distractor says he beat Sinner **to reach** the semifinal. The supported
condition additionally contains an explicit passage about outlasting Frances
Tiafoe in the semifinal. Thus the observed mistake is event-relation confusion,
not simply unwillingness to abstain. Stronger generic instructions did not fix
it. Attention, neuron attribution and a causal mechanism have not been measured.

This is one exposed example, not a sufficient training/evaluation set. Training
on it and then celebrating its correction would repeat the earlier fact-edit
ceiling problem. Even treating the venue specificity and cut-off response as
additional failures gives three affected questions, below the frozen four.

## Question-only control and limitations

The 22 question-only responses have a different contract: use model knowledge.
Only one completed answer agrees fully with the archived reference, two return
UNKNOWN, and the rest include disagreements or unsupported denials. Several
questions are undated/time-sensitive, so this is not a present-day factual
accuracy benchmark. The game-release answer matches the reference month/day
but supplies a wrong year: another concrete reason not to use substring scoring.
The documents clearly help on these development examples; ignoring them is not
a solution here. Unexpectedly weak knowledge-only outputs warrant checking
inference/model conditions before interpreting them as general model capability.

Eleven of 110 outputs hit the frozen 48-token cap: two supported responses for
the same question and nine question-only responses. They remain incomplete in
this report; we did not lengthen generation after seeing them and call those
results the frozen run. No missing-evidence response was truncated.

The 22 questions were selected by output-independent ordering and document
review, but remain a small curated development sample, not an untouched test
population. Raw source documents and predictions remain private research
artifacts under the source dataset's noncommercial license. No claim of
selection benefit, behavioral improvement or paper reproduction follows.

## Consequence

Close this qualification without fitting adapters or searching another generic
abstention threshold. The useful lead is the specific event-relation confusion,
which would require independently varied examples and transfer tests to establish
a reusable weakness. Separately, the very weak knowledge-only control makes a
small inference sanity check more justified than treating these outputs as a
normal capability baseline for a new method. Neither is a success claim.
