# Does the event-relation failure repeat with unfamiliar entities?

The [retrieval baseline](../retrieval_evidence/complete_v1/REPORT.md) had one
persistent mistake: the model answered whom a player defeated **to reach** the
semifinal when asked whom he defeated **in** the semifinal, even with the correct
passage present. The [precision check](../inference_sanity/complete_v1/REPORT.md)
did not change that answer. This bounded probe asks whether a related mistake
repeats in short, explicit contexts without reliance on factual recall.

The 96 inputs cross four families with three fictional name sets, swapped
candidate roles, and four evidence conditions: target passage first, target
passage last, target only, and target absent. These are twelve family/name cells
and four templates, not 96 independent tasks. Target-only controls distinguish
basic extraction failure from distractor interference. Missing-target controls
separate unsupported guessing from discrimination between competing passages.

Every scenario and label is authored before inference. The complete source is
in `prepare.py`; a literal fixture checks the reference roles, name permutation,
absence of the target name in missing evidence, absence of the distractor in
target-only evidence, and preservation of documents/labels under reordering.
The existing unchanged-model worker is reused; there is no training or new
model implementation. Max generation is 96 tokens; no prompt truncation.

The prospective qualification is at least three family/name cells across at least
two families with a wrong supported answer, while at least 22/24 target-only
answers remain correct. This merely identifies a repeated development weakness.
All full answers require review, with names in possessive/full sentences accepted
where correct. A repeated template error is not proof of a unique neuron circuit,
an adaptation advantage, or natural-language generalization.

If these short fictional cases are solved, that narrows the observed error to
conditions absent from this probe; it does not distinguish factual associations,
context length, distractor richness or their interactions. Do not respond by
adding arbitrary noise until the benchmark fails. If the qualification passes,
first compare ordinary LoRA under the same labels before claiming value from a
selector. A future method still needs natural inputs, fresh domains, preservation,
multiple seeds and actual cost comparisons. The goal is not redefined around
passing a synthetic task.

## Completed result

The [reviewed result](complete_v1/REPORT.md) is 75/96, with all 24 target-only
controls correct. The frozen development qualification passes. A learning pilot
is warranted; no adaptation or selected-neuron benefit has yet been measured.
