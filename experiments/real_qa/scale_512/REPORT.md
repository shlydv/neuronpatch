# Fourfold larger training set: improvement at one rate, substantial interference remains

Completed both prespecified one-epoch, 512-example all-MLP LoRA fits. Model,
rank, objective, replay policy and validation population match the 128-example
reference. The larger fit uses 256 rather than 64 optimizer steps and 91 rather
than 20 preservation anchors; this is not an equal-compute data-diversity experiment.

| Train size / rate | Supported EM /128 | Supported refusals /128 | Correct impossible refusals /128 | Retained base-correct answers /45 |
|---|---:|---:|---:|---:|
| Base | 45 | 13 | 34 | 45 |
| 128 / .0001 | 40 | 47 | 81 | 34 |
| 128 / .0003 | 43 | 33 | 62 | 36 |
| 512 / .0001 | 31 | 75 | 107 | 26 |
| 512 / .0003 | 46 | 56 | 95 | 37 |

At .0003, more data/steps/anchor coverage improves supported exact match and correct
refusals while slightly improving retention of the base's known-correct answers.
But seven base-correct answers become refusals, and another loses exact match.
The overall supported exact-match gain over the base therefore hides eight lost
matches offset by nine newly exact matches. Those nine require semantic review:
some may have been correct paraphrases before training. Supported F1 is .422 versus
.458 for the base. This is a trade-off, not unqualified improvement.

At .0001, the larger fit primarily strengthens refusal and worsens supported
answers. Do not average away this learning-rate sensitivity or select only the
better-looking rate. Both remain in the curve. On the recurring diagnostic pair,
.0003 still answers RNA replicases to both presence and absence questions; it does
recover Hoover but unnecessarily refuses the shy/reserved Virgil question.

The 2048-example comparison remains pending. There is no neuron-localization result
here and no final-confirmation inference. Raw predictions, exact sources, hashes,
loss histories and paired known-correct retention are retained. The next conclusion
must incorporate the full planned curve, not just its most favorable checkpoint.

## Semantic audit of the .0003 checkpoint

`semantic_review.json` records all 17 supported exact-match changes, with full
contexts, outputs and source hashes. This is assistant qualitative review, not
blinded adjudication; automatic scores and dataset labels remain unchanged.
Of nine gains, four only shorten or normalize already-correct answers, one is an
ambiguous container alternative, one adds a numerical qualifier, and two clearly
correct the answer. The remaining gain exposes a question/gold mismatch: “Where
was the UCR created?” has “1920s” as its reference. Replacing UNKNOWN with a decade
earns exact match but does not correctly answer that location question.

Of eight losses, seven are correct answers replaced by UNKNOWN. The eighth drops
“Jr.” from an agent's otherwise correct name. Moreover, nine inspected nonexact
but valid baseline answers become UNKNOWN without changing their zero exact-match
scores: these include LPC for linear predictive coding, 731 for the team count,
and sentence-form answers containing the requested information. This subset is
not a complete semantic evaluation and must not be used as an overall accuracy
estimate. It demonstrates that preservation damage extends beyond lost exact
matches. Short-span compliance and answer correctness need separate accounting.

The preservation anchor rule also uses exact match, so it can omit valid baseline
answers from protection. That is a concrete hypothesis for an objective/data
improvement, not yet a demonstrated explanation or fix. Current runs retain the
original rule to preserve their comparison. Any broader anchor policy must be
defined using training data only and compared against the same policy with random
selection and LoRA; validation review must not silently become training labels.
