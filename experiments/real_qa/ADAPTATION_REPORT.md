# Output-aware paired selection does not yet beat random row updates

Exploratory development comparison, 2026-10-19. Eight fits completed on local MPS;
confirmation remains unevaluated. The recipe and limits were recorded before fits
in PROTOCOL.md. Raw predictions, selection scores and losses are in `adaptation/`.

| Method | LR | KL | Supported EM /16 | Supported F1 | Supported refusals /16 | Correct impossible refusals /16 |
|---|---:|---:|---:|---:|---:|---:|
| Base | — | — | 4 | .467 | 1 | 3 |
| Selected rows | .0001 | 1 | 5 | .466 | 5 | 5 |
| Selected rows | .0003 | 1 | 2 | .145 | 10 | 9 |
| Random rows | .0001 | 1 | 3 | .405 | 2 | 4 |
| Random rows | .0003 | 1 | 6 | .529 | 2 | 6 |
| LoRA | .0001 | 1 | 3 | .395 | 3 | 5 |
| LoRA | .0003 | 1 | 2 | .180 | 11 | 10 |
| Selected rows | .0001 | 0 | 5 | .466 | 5 | 6 |
| Selected rows | .0003 | 0 | 2 | .145 | 10 | 10 |

## What the actual outputs show

At LR .0001 with KL, selected rows gain two supported exact matches: RNA replicase
becomes RNA replicases, and a Hanover Zoo sentence becomes the short span. Both
were already substantively correct. The correct Hoover answer becomes UNKNOWN.
Two impossible questions are newly refused (Pollio's residence and the number of
live-bearing mammal species). Four additional supported questions are refused,
including previously wrong answers and a previously supported herpes-virus example.
The presence/absence enzyme pair still receives essentially the same answer.
Thus a net EM improvement does not mean improved evidence discrimination.

Random rows at .0003 gain two wording matches and switch the reusable-bottle answer
to the dataset's refillable-jug answer (the original was a source-supported but
question-ambiguous alternative). They also newly refuse three impossible questions.
They still break Hoover and do not fix the enzyme distinction. This is a better
observed trade-off than selected rows, not proof of a generally better method.

At the higher rate, selected rows and LoRA become broadly reluctant to answer.
Their higher correct-refusal counts are accompanied by substantial supported-answer
loss. A single aggregate accuracy would obscure that behavior.

## What this says about the synthesis

1. The paired output-aware score has not earned its discovery cost. Pairing passages
   removes one topic confound; it does not make a last-token output-alignment score
   identify parameters that can learn a conditional relation check. First-token
   scoring and omitted downstream nonlinear computation are concrete limitations.
2. KL on 20 exact-correct training anchors did not improve supported validation
   metrics for the selected method at either tested rate. The KL was active on 18
   batches, so it was not simply skipped. This does not rule out different anchors,
   coefficients or preservation contexts; it rules out claiming this setting fixes
   generalization. With KL, one fewer impossible question was refused at each rate.
3. The prior synthetic up-row benefit does not establish safe transfer to real QA.
   Row location, localization score and objective must earn their place separately.
4. A better method must show conditional correction, rather than only identify
   neurons that can move the overall answer/refusal tendency. Candidate discovery
   should be tested for actual corrective-versus-damaging update effects, not accepted
   solely because activation and output alignment appear interpretable.

## Next diagnostic before adding another selector

Evaluate these saved adapters on their training and calibration examples to separate
failure to learn the relation from failure to transfer it. Inspect paired refusal
margin changes and anchor retention, including semantically correct non-EM answers.
A narrow supervised anchor set may miss important preserved behavior; conversely,
preserving every base output would freeze real mistakes. Resolve this with evidence
before expanding the objective or using a learned selection criterion. A useful
selection criterion would need to predict improvement on separate calibration
pairs and retain independent confirmation, with matched discovery budgets.

One training seed, one random support seed, 16 validation pairs, two learning rates,
and unequal scalar parameter counts do not support a superiority claim. No published
method was faithfully reproduced here. No candidate is promoted as the default and
no confirmation set is spent on these weak candidates. The broader goal remains open.
