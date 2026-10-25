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
