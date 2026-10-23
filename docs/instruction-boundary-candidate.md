# Candidate behavior task: preserve the task across an instruction boundary

24 October 2026. The math slice is retired without training. This note selects
the next development setting after source inspection; no new GPU job or model
result is claimed. The goal remains a useful sparse adaptation advantage.

## Why this is more relevant than another elementary skill

A user asks a model to answer a question from a document. The document contains
both useful evidence and an instruction to do something else. The desired
adaptation changes which instruction the model follows while retaining its
ability to read the document and answer normally. This is a behavioral change,
not an attempt to teach arithmetic the model already knows. Whether our pinned
model actually needs this adaptation still requires a controlled baseline.

## Source evidence that changes the design

[SecAlign](https://arxiv.org/html/2410.05451) trains preference pairs: the chosen
response follows the legitimate instruction and the rejected one follows the
injection. Its published LoRA setup targets q/v attention projections, rank64,
with a substantially larger hardware budget. Our small MLP-only SFT runs cannot
be interpreted as a reproduction or refutation. Comparing sparse SFT to LoRA
preference training would confound representation and objective.

The [pinned training source](https://github.com/facebookresearch/SecAlign/blob/3fc9c9c3244a21512f5a1f3859a9858090711f0e/align.py)
constructs desired/undesired outputs from clean instruction examples and delegates
optimization to TRL. It is CC BY-NC4.0; no upstream source is copied into this MIT
library or executed. The principle of contrasting competing responses can be
implemented independently with attribution. This does not make preference tuning
a new method.

[BIPIA](https://github.com/microsoft/BIPIA/tree/a004b69ec0dd446e0afd461d98cb5e96e120a5d0)
provides paired document tasks and injected text. The inspected QA, email and
table builders place external context inside the **system message** by default.
Their alternative branch combines it with the user prompt. A test of lower-trust
document content must explicitly define that boundary instead of inheriting the
default and calling it a faithful application of modern chat hierarchy.

The training-only source inventory is in
[source_audit.json](../experiments/instruction_boundary/source_audit.json):

| Slice | Questions | Distinct normalized contexts | Unknown labels |
|---|---:|---:|---:|
| Email | 50 | 45 | 33 |
| Table | 900 | 617 | 0 |

The email builder checks the misspelling `unkown`, while downloaded labels use
`unknown`. This is another reason not to import its response builder blindly.
The table slice has up to four questions per table, so row-wise random splitting
would leak documents across banks. Its longest raw context is7021 characters;
token lengths still need checking. No official test data was downloaded.
The dataset's source provenance and licensing remain separate from BIPIA's code
license; do not automatically redistribute third-party contexts under our MIT
notice. Source files are held privately as research artifacts.

## Chosen development design

Use table QA with disjoint documents across training, calibration and development.
Keep the native model chat template fixed. The system message defines the task
and states that supplied documents are data. A user message supplies the question
and explicitly quoted document. Do not put external content in the system role.
This is a declared adaptation of the benchmark, not its original protocol.

For every development question compare the clean document with the same document
containing a harmless instruction to output an unrelated, per-example marker.
Use several predefined instruction families, not an adaptive attack search.
Add a direct-user control that legitimately requests the same marker, and a
quotation control that asks about instruction text as data. No real tools,
accounts, private records or harmful actions are involved.

The central endpoint is preservation of **correct legitimate answers**, paired
clean-to-injected. A lower marker rate is insufficient: refusals, broken reading
comprehension and generic answers also suppress markers. Report direct-user
compliance separately so a model that simply refuses the marker cannot win.
Do not count hundreds of payload variants as hundreds of independent documents.
Reference aliases and ambiguous questions require review before success claims.

Baseline qualification must establish correct clean answers that become wrong
under injection, rather than selecting questions the model already cannot answer.
Freeze sampling and the error requirement before inspecting outputs. Neither
attack construction nor a public benchmark guarantees enough headroom.

## Method hypothesis, with the components kept separate

FiNE motivates matching localization to the exact change; NeST motivates testing
structured trainable capacity; SafeNexus motivates checking intervention effects
against normal behavior. None licenses combining their headlines into a recipe.

Here, a candidate selector should predict the update's effect on **competing
responses across the trust boundary** and on legitimate controls. Match document
content when forming contrasts to reduce topic confounds. Use the actual-update
observer to test that prediction before treating its ranking as useful. This is
a hypothesis, not a unique causal localization claim or a security guarantee.

First hold objective and data fixed across ordinary LoRA, random supports and
guided supports. Separately compare complete-response SFT and a properly
implemented preference objective; give LoRA the same objective opportunities.
Any gain shared by all representations is objective/data improvement, not a
neuron-selection advantage. Attention-capable LoRA is an essential baseline in
this setting; artificially restricting it to MLPs would miss a relevant control.
Clustering adds a further hypothesis and should not be bundled into the first
comparison. A successful small study would still require held-out instruction
families, another document task, seeds and measured discovery/training costs.

Next work: prepare the document-disjoint data and role/label controls, check their
tokenization, then freeze one bounded baseline qualification. No training or
new named method is justified by this source audit alone.
