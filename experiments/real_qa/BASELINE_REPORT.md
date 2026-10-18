# Real passages reveal both genuine errors and scoring artifacts

2026-10-19. Unedited pinned Qwen2.5-0.5B-Instruct; no new adapter trained.
Article-separated paired SQuAD 2.0 samples, full contexts, all supplied aliases.
Confirmation predictions have not been generated. See PROTOCOL.md.

| Development split | Supported EM | Supported token F1 | Unnecessary refusals | Correct impossible refusals |
|---|---:|---:|---:|---:|
| Train | 20/64 | .439 | 12/64 | 17/64 |
| Calibration | 8/16 | .720 | 1/16 | 0/16 |
| Validation | 4/16 | .467 | 1/16 | 3/16 |

One training completion hit the 32-token output cap; none in validation or
calibration. These are automatic extraction metrics, **not semantic accuracy**.
The entire baseline took about 25 seconds after loading on local MPS.

## Manual validation inspection changes the next experiment

Reading the passage, question and output for all 12 supported EM mismatches found:

- Six substantively correct responses lose exact match for wording: the calendar
  explanation, singular RNA replicase, shy/reserved without "extremely", full-sentence
  Hanover Zoo and US Constitution answers, and "centuries" instead of "century".
- One answer (herpes virus) gives a supported example rather than the requested
  general class (some viruses); count this separately as scope ambiguity.
- One answer (reusable bottle) is explicitly listed as an alternative in the
  passage but the question's "better than a bottle" wording is ambiguous. Do not
  silently add it as a gold alias.
- Four are substantive problems: unnecessary TRON refusal; mammals instead of the
  named cynodont group; repeating "live animals" instead of identifying microbes;
  the Ark-of-Noah answer instead of the stated Marian foreshadowing. The final
  question itself is awkward, so preserve its provenance for review.

These are assistant qualitative annotations, not independent blinded judgments;
raw automatic scores remain unchanged. No post-hoc aliases are applied to training
or confirmation. IDs and explanations are in `validation_review.json`.

The enzyme pair is particularly useful: changing "presence" to "absence" leaves
the response at RNA replicase. The passage supports presence only. Other impossible
questions alter dates, entities or relations. This task demands checking whether
the *relation in the question* is supported, not merely detecting a relevant topic.
We have not yet established the neural mechanism behind these errors.

## Implication for the combined method

A generic answerable-versus-missing contrast can reward topic or response-style
signals. The same-paragraph pairs reduce topic confounding and can support a more
specific probe or candidate score. That hypothesis still requires causal tests.
More importantly, a patch might improve exact match solely by shortening answers.
Report genuine correction, verbosity/format changes and new refusals separately.

Next: audit training preservation labels, using exact-correct cases as a conservative
anchor while explicitly acknowledging missed paraphrases; compare that policy
against broader reviewed anchors. Use the paired examples to investigate conditional
support rather than optimize a blanket UNKNOWN direction. Keep random rows and LoRA
as controls, and use real validation failure cases for development before opening
confirmation. No evidence yet establishes superiority over any published method.
