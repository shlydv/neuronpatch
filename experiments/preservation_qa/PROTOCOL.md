# Preservation follow-up, fixed before outcomes (2026-10-19)

Hypothesis from pilot: activation contrast identifies group differences, not low-damage
update locations. All seven new errors were UNKNOWN on supported answers. Six were
exhibits and one an expedition, but this is descriptive, not causal evidence.

Keep the original model revision, layers, 16 columns/layer, 32 steps, batch size 4,
optimizer, prompt instruction, and greedy decoding. Fresh disjoint synthetic entities:
96 train, 24 calibration, 48 validation, 96 final questions. Test families are
clinics and vessels; training families are depots and seminars. Validation includes
additional role fields not used in training, with separate entities. No old final
test is used for model selection. The test remains synthetic record extraction.

Methods: original activation selector; gradient sensitivity selector; three random
seeds; same-layer rank-2 LoRA. All methods receive learning rates 0.0003 and 0.001,
identical order, data and training budget. The training objective is unchanged
(response-token cross entropy). This isolates selection and validation-policy changes.

New selector: for each calibration example compute the full teacher-forced response
loss gradient with respect to each candidate down-projection weight. Average squared
gradient norms per column separately for missing and supported questions. Rank
correction_energy / (preservation_energy + mean(preservation_energy) + 1e-12).
This is a sensitivity heuristic, not a causal discovery algorithm. Squaring before
averaging avoids cancellation across examples. Record discovery overhead. No extra
training or selector hyperparameter search.

Select a candidate on validation only if supported accuracy is no more than 5 points
below base, missing accuracy improves by at least 10 points, and arithmetic loses
at most 1/16 items. Choose eligible candidate by balanced accuracy, then supported
accuracy, then lower rate. If none qualify, choose the unmodified base (explicit
no-edit outcome). Also report all candidates' validation results. Freeze choices
before evaluating final data. Fresh arithmetic inputs are shared across validation
and final monitoring; therefore this is a selection sentinel, not an untouched
capability test.

Final gate is the same >=10 point missing improvement and <=5 point supported drop.
Report selected-versus-random mean and LoRA; no claims from a best random seed.
An improvement over original selection requires >=5 points balanced accuracy over
that method under the same validation policy; useful superiority requires also
meeting the preservation gate and beating random mean by >=5 points. A fallback
is safe but is not a successful edit. One small task cannot establish robustness.

Check output artifacts by reloading before final evaluation. Save data, raw outputs,
source/data hashes, indices, gradient statistics and all validation choices. Stop
after this experiment without tuning against final outcomes.
