# Cross-context distillation candidate — 25 October 2026

Status: prior-art check and CPU loss primitive complete; no GPU submission.
The goal remains a useful sparse-adaptation advantage, not merely a new defense.

## Prior art changes the claim

[RAISED, section 5.2 and appendix C.5](https://arxiv.org/html/2610.06401v1)
uses frozen clean-trajectory distributions to supervise both clean and injected
contexts. It stores top-256 probabilities plus residual mass, trains all-linear
LoRA, and reports 10,000 retained generated trajectories, batch 16, one epoch,
learning rate 5e-5 and injection ratio 0.5. This closely overlaps our proposed
next objective. Its linked GitHub repository, `orailix/RAISED`, returned HTTP404
through the GitHub API on 25 October; no source reproduction is claimed.

[SecOPD](https://arxiv.org/html/2608.21500v1) instead samples student responses
under attack and scores those exact tokens with a frozen clean-context teacher.
The released full-response recipe turns sampled reverse-KL into token advantages.
We inspected commit `571502a2a315c4b8820dd878d4569e2a2222cb88`, especially
[the active override](https://github.com/pppyb/SecOPD/blob/571502a2a315c4b8820dd878d4569e2a2222cb88/training/tinker/overrides/tinker_cookbook/recipes/distillation/pi_defense_full_response_kl.py).
It overrides the older reasoning-specific path; reading that older file alone
would misidentify the released objective. Its paper configuration uses Qwen3.6-27B,
rank128, groups64, one rollout per group and 150steps. We have not reproduced it.

Neither paper establishes that our 31-document, rank-one setup should succeed.
Cross-context KL is existing work, not our novelty. A compact implementation can
be a baseline; the library contribution still needs practical quality, retention
or measured cost gains against random supports and LoRA.

## What is implemented

`experiments/instruction_boundary/cross_context_retention.py` reuses the existing
full-vocabulary KL and response cache. It aligns prediction positions by their
continuation offset despite different prompt lengths. The reference stays bound
to its clean tokenized trajectory and base identity. It rejects changed teachers,
response-token mismatches and gaps in supervision. It does not infer which prompt
is trustworthy or whether a teacher answer is correct. Callers must pin the same
vocabulary and supply the correct clean/student pairing.

Two new CPU tests verify the scalar loss and gradients against an independent
calculation and reject corrupted alignment. All six focused tests pass. This is
teacher-forced forward KL, not RAISED's compressed KL or SecOPD's sampled objective.
No method-equivalence or benchmark replication claim follows.

## Before another run

Use source-reviewed teacher trajectories only from training/calibration documents.
Earlier hard-target experiments retained teacher mistakes; do not repeat them.
Do not reuse outputs when question wording or prompt identity changed. If filtering
incorrect teacher answers reduces the bank, report every exclusion and use the
same bank for controls; a teacher-preservation test cannot claim new factual skill.

Full FP16 references cost 303,872 bytes per response position with vocabulary
151,936. A 31-example bank at 64 positions each would use about603 MB, versus the
last control's36 MB for119positions. Actual lengths must be measured before fixing
a cache cap. Reuse a clean teacher cache across paired attacked variants, but do
not hide teacher generation/scoring and review costs. Top-K compression would be
a separate approximation requiring its own gradient/error checks.

Next needed evidence: inventory identity-matching existing teacher trajectories,
verify correctness/completion and measure actual cache cost. Then specify a bounded
objective comparison, keeping parameter support fixed. Its prediction is reduced
attack-marker compliance while retaining clean answers. Only after a useful
baseline exists should selected independent neuron rows compete against matched
random rows and LoRA on fresh transfer and unrelated-task retention.

## Teacher inventory and review checkpoint

The saved bank has 23/31 training and 12/15 calibration trajectories whose prompt
tokens match the current questions, whose decoded text matches their tokens, and
which end with EOS. Eleven changed prompts cannot reuse
their old responses as exact clean-context trajectories.

All 35 matching trajectories are now source-reviewed. Four were rejected: two training
answers select the wrong producer's track and miscount songs under two minutes;
two calibration answers miss consecutive broadcaster seasons and select the
wrong minimum-temperature month. The private decisions bind exact task/teacher
hashes. The other 31 were accepted: 21 training and 10 calibration. One calibration
response initially says five away games, then explicitly recounts and corrects
the answer to six. Its correct final answer and source-supported enumeration are
accepted with that caveat recorded; the trajectory is not silently rewritten.
These are assistant judgments, not independent human review.

`teacher_review.py` now requires one explicit accept/reject decision for every
inventory item, binds decisions to the source document, question, prompt tokens
and complete response, and preserves exclusions in a receipt. Changed content,
missing decisions, duplicate documents and cross-split overlap fail validation.
The initial partial review correctly failed with 31 pending; the complete review
passes. Eleven focused tests passed across review, cross-context loss and response
retention. This gate is not
yet connected to a GPU worker; it cannot establish semantic correctness itself.

Do not replace rejected responses with reference strings and call those
self-generated clean teacher trajectories. A corrected-target objective is a
different control. Filtering must use identical accepted banks across comparator
arms, and the original unfiltered evaluation population must remain visible.

`prepare_cross_context.py` verifies the archived teacher files against the recorded
parent hashes, checks model/revision consistency, checks every retained trajectory
against its archived original, and runs the complete-review gate before exporting
a bank. The current bank passes with recorded base fingerprint
`3b369c58239c915efbf22934957e32f5134a697729ccad78f2d8e7b29b76ee5c`.
Five additional tests check valid preparation and corrupted source, inventory,
revision and ambiguous origin. Runtime must independently verify the loaded base,
tokenizer, clean prompt IDs and termination before caching logits.

The [preparation summary](../experiments/instruction_boundary/cross_context_preparation_v1/summary.json)
pins the review and inventory. Accepted training responses have 572 positions,
requiring 173,814,784 bytes for full FP16 reference logits. Calibration has 562
positions and would require 170,776,064 bytes if cached. Neither includes model,
activation or softmax workspace memory; cache only the bank needed for fitting.

## Next comparison, before sparse selection

The bounded comparison should train two fresh, identical LoRA adapters on the
same 21 accepted documents and the same trusted/untrusted templates. Both retain
the same clean-distribution KL term and trusted-instruction targets. Change only
the untrusted-context target: hard teacher-response NLL versus clean-context
distribution KL along that exact response. This separates the soft target from
teacher filtering, response length and parameter support. The previous 31-document
short-target adapter is historical context, not the causal control.

The prediction is fewer explicit attack-marker responses without more lost clean
answers or less legitimate-instruction compliance. Failure on either side rejects
this candidate at the tested budget; it does not trigger another selector sweep.
Report paired outcomes, unfinished answers, step/token exposure, all fitting and
cache costs, and exact export/reload behavior. Existing exposed confirmation
documents can diagnose this prediction but cannot establish fresh transfer or
library superiority. A successful development result still needs fresh examples,
matched sparse/random comparisons and the broader completion requirements.

This comparison is a design decision, not a frozen runnable protocol or a
submitted job. Worker integration and runtime bounds remain to be completed.
