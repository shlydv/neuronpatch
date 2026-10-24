# Next baseline: distinguish coverage from objective

The [completed supervision test](../experiments/instruction_boundary/complete_supervision_v1/REPORT.md)
restored legitimate requests but weakened unfamiliar attack resistance in every
representation. The learned-mixing experiment remains deferred. This document
prepares an objective control; it does not freeze or launch another GPU run.

## Evidence from actual training inputs

The saved training bank contains32 injected,32 clean,32 full-document direct and
32 quoted examples. All instruction-bearing examples use **one redirect wording**.
Calibration repeats that wording on16 different documents. Thus low calibration
loss measures new-document generalization for a familiar instruction, not the
cross-wording distinction that fails in confirmation.

[preference_preparation.json](../experiments/instruction_boundary/preference_preparation.json)
records a CPU check on the exact saved banks. The64 training and32 calibration
preference pairs use the same prompts and chosen responses as the current
injected/direct-full supervision, byte for byte. No source reference answers or
confirmation responses were used. Clean teacher responses can contain factual
mistakes; preference construction does not certify those answers.

## Why compare a different objective

[SecAlign](https://arxiv.org/html/2410.05451v3) trains a reference-relative
preference between the original task's answer and the injected task's answer.
Its [official code](https://github.com/facebookresearch/SecAlign/blob/main/align.py)
constructs these alternatives from different instruction examples and delegates
training to TRL. The published recipe uses varied natural instructions, largely
straightforward injection plus some completion spoofing. It is much richer than
our single marker request. Its code uses rank64 q/v LoRA, unlike our rank1
all-projection adapter. Neither its results nor configuration are reproduced by
adding one loss function here.

SFT already competes with other tokens through softmax; it is incorrect to say
that SFT never penalizes wrong answers. DPO explicitly compares the two response
sequences relative to a frozen model. It can improve that relative margin without
necessarily increasing the absolute probability of the desired response, so
free-running answers and legitimate compliance remain the outcome measures.

For each matched pair we need both directions:

- Instruction inside the untrusted document: prefer the source answer to the
  injected instruction's response.
- The identical instruction outside the document: prefer its legitimate response
  to the now-obsolete source answer.

This reverses which output is correct and prevents a global output blacklist from
satisfying both examples. This is ordinary contrastive supervision with known
preference-optimization prior art, not a novel neuron-selection method.

## Implemented and checked

`experiments/instruction_boundary/preference_pairs.py` constructs the two
contexts and preserves cached teacher identity. It rejects missing membership,
duplicate teachers and conflicting marker/answer identities, and excludes incomplete
teachers.
A supplied membership is not a magical leak detector: the runner must bind it to
the frozen training split. Templates and any new confirmation split still need
separate review and freezing.

`preference_objective.py` computes full-response sequence log probability with
the existing validated causal mask, including EOS. Standard DPO uses sums,
not mean-token probabilities. It requires detached reference scores and uses
stable log-sigmoid arithmetic. It is experiment-local, not a new public default.

Thirteen focused CPU tests pass, including existing causal-mask checks, exact
trust-location reversal, reference-label independence, finite-difference loss
gradients, and gradient agreement with explicit sequence scoring through an
actual tiny Qwen/LoRA model. No downloaded full-size model was run on the Mac.
No test establishes large-model transfer, reference-cache provenance or GPU
memory suitability. Those remain runner/integration requirements.

## Experiment decision before launch

Separate data coverage and objective in a small factorial comparison on the same
LoRA representation: narrow versus diverse instruction training, each using SFT
or DPO. Chosen targets, contexts and preservation data must match within each
objective comparison. Report token/forward-pass cost; equal optimizer steps are
not equal compute. Do not compare a retuned DPO model with an untuned SFT model
and attribute the difference solely to the objective.

The diverse data should contain genuinely different requested actions and output
content, including non-marker answers; a list of paraphrases of “return MARKER”
is an inadequate substitute. A [source-reviewed table-task bank](boundary-training-label-audit.md)
is now prepared with matched competing questions and non-marker responses. It
remains limited to table QA; it is not broad instruction coverage. The existing
marker constructor is retained for the narrow control.
Use trusted counterparts for all requests and withhold instruction wording or
families independently from documents. Prefer the public natural-task construction
above over another ad hoc pile of template variants. Preserve clean and quoted
instruction-reading cases, and evaluate complete generated answers.

Before submission: construct and inspect the richer labels, bind source/model and
reference-score hashes, freeze fresh evaluation membership and absolute utility/
robustness criteria, and test the actual optimizer/export path on a tiny model.
No new run is authorized by this document alone; use the user's existing scope
and resource constraints when deciding to execute.

A working LoRA result supplies a behavior worth making cheaper. Then test sparse
selection and sharing against that competent baseline with matched task coverage,
additional seeds, unrelated-task retention and full costs. A successful preference
baseline alone would not achieve the neuronpatch goal or establish novelty.

## Integrated optimization and export check

`preference_training.py` now caches frozen reference sequence scores, binds them
to tokenized-pair identities, and runs common Adam/preservation schedules. DPO
recomputes each response separately with the exact scalar chain-rule coefficient,
retaining one model graph at a time. This trades memory for compute: four task
forwards per DPO update versus one for SFT, plus one clean-preservation forward
for both. Reference scoring costs two additional forwards per unique pair.
Nonfinite gradients retry the same samples at a lower scale without an Adam step.

`preference_experiment.py` runs all four coverage/objective arms from independently
initialized adapters on the same fingerprinted base. It rejects conflicting data
identities and existing output directories, verifies base restoration after every
arm, and evaluates reloaded artifacts after exact two-response score checks.
The evaluation callback must save full responses; no metric or checkpoint choice
is hidden in this orchestration layer.

Fifteen focused CPU tests pass, including the complete four-arm optimizer/export
path on a randomly initialized tiny Qwen model, direct-gradient and Adam-update
agreement, unequal response lengths, nonfinite retry, and base restoration.
These establish implementation behavior, not full-model quality or T4 feasibility.
Fresh held-out evaluation, frozen protocol, model-loading/bundling and GPU runtime
checks remain before submission. No new GPU job has been launched.

## Fresh evaluation and bounded protocol

The frozen [source review](../experiments/instruction_boundary/preference_confirmation_v1.json)
allocates32 documents excluding all400 previously allocated boundary documents.
Two source-invalid questions are excluded before model outputs and six are
clarified;30 documents remain. The360 cases include clean QA, three marker-attack
families with full-document legitimate counterparts, quoted-marker reading, and
competing real table questions in both trust locations. Competing questions use
both familiar and independently held-out wording on the same donor assignment.
All source review is by the assistant, not an independent human.

[Protocol v1](../experiments/instruction_boundary/preference_protocol_v1.json)
fixes186 updates, a common rate, rank-one all-projection LoRA, and final-checkpoint
reporting. This covers every diverse pair once and every narrow pair three times;
equal steps still do not equal compute or unique-example exposure. Calibration
remains diagnostic. Absolute quality and preservation gates are frozen before
outputs, alongside paired improvements and failure analysis. One seed/rate is a
bounded diagnostic, not an exhaustive comparison of objective potential.

No new GPU run has been submitted. Hardware execution/packaging remains pending.
Passing this table-task evaluation would not establish unrelated-task retention
or the project's requested sparse-adaptation advantage.
