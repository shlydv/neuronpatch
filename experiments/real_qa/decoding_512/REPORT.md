# Neutral repetition penalty fixes some cases, not the preservation trade-off

Completed all five evaluations with process exit 0. No weights changed. Each uses
the same 256 development questions and greedy generation, changing only inherited
repetition_penalty=1.1 to an explicit 1.0. The base is reevaluated under the same
policy as both rates of the exact-anchor and expanded-anchor models. Original
predictions remain intact; source/checkpoint hashes and complete generation
configuration are recorded in summary.json. Confirmation remains unused.

| Model / rate | Supported EM default→neutral /128 | Unnecessary refusals default→neutral /128 | Correct missing refusals default→neutral /128 |
|---|---:|---:|---:|
| Base | 45→47 | 13→9 | 34→30 |
| Exact anchors / .0001 | 31→31 | 75→70 | 107→101 |
| Exact anchors / .0003 | 46→47 | 56→52 | 95→92 |
| Expanded anchors / .0001 | 30→32 | 73→67 | 103→98 |
| Expanded anchors / .0003 | 39→43 | 63→59 | 100→97 |

The penalty contributes to refusals in both directions. Removing it recovers some
supported answers but also elicits answers to missing-evidence questions. The
expanded-anchor model still trades fewer supported exact answers for more correct
missing refusals than the exact-anchor model at the higher rate. Coverage is not
vindicated by correcting the decoder comparison.

## Output inspection

For the higher-rate expanded model, neutral decoding recovers answers such as
80% copper reuse, 731 teams and LPC; the latter two remain nonexact despite being
supported. It also earns a spurious exact match by answering the malformed “Where
was the UCR created?” question with 1920s. Other gains include a correct time period
for international political-party organization and the purpose of ironing.

New missing-evidence answers include 224 deaths for Operation Infinite Reach and
8.6 million for total chronic TB cases. These cases require relational reading of
the passage; selecting a nearby number is not support. The base likewise changes
an appropriate refusal to 0.4–1.0% for a mercury question in a copper passage, and
answers 2,500 rather than 800 for a college-of-music question. Lower refusal rates
are not equivalent to better answers. These examples are qualitative diagnostics,
not a complete blinded semantic accuracy estimate.

## Consequence

Generation settings are part of the intervention/evaluation contract. Future
experiments must record effective decoding policy explicitly rather than equating
greedy generation with raw argmax. Neutral penalty is a useful controlled reference
for passage extraction, not a claimed universally superior decoding policy.

The same-checkpoint audit previously established five recoveries on selected
training examples. This broader result limits that finding: it is a real mechanism
but not a general solution to interference. Raw-logit preservation failures still
need investigation. The next diagnostic measures per-token KL on all 115 training
anchors and compares the first-token error with the response average before
changing the loss or neuron-selection method. No superiority claim follows from
this decoding correction.
