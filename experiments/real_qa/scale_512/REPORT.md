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
