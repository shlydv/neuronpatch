# Loss-isolation diagnostic — completed 25 October 2026

**Both training signals independently damage clean QA at this recipe. Removing
preservation makes the task-trained model worse.** This is a causal loss-term
ablation on exposed development cases, not fresh confirmation or a new method.

| Condition | Clean | Authority injection | Competing injection, held-out wording | Legitimate competing task, held-out wording | Base-correct clean losses |
|---|---:|---:|---:|---:|---:|
| Untouched base |23/30|0/30|2/30|19/30|0|
| Previous combined SFT |20/30|19/30|18/30|20/30|6|
| Task only |17/30|17/30|18/30|18/30|7|
| Short-reference preservation only |21/30|2/30|2/30|18/30|5|

Task-only reproduces **all six** combined-fit clean regressions and introduces
one further regression. Preservation-only reproduces **five of the same six**.
Thus preservation training is sufficient to damage those five answers, but it
is not the sole cause: task training alone is also sufficient. At the final
checkpoint, combining the losses gives more correct clean answers and fewer
regressions than task-only. Removing the auxiliary loss is not a repair.

All trained conditions give short answers: mean clean lengths are 4.03 task-only,
4.27 preservation-only and 4.13 combined, versus 27.03 for base. This association
does not establish that brevity causes the errors. Both objectives favor short
reference completions and share the adapter, learning rate and optimization.
An ablation establishes the effect of removing a term at this recipe; it does
not identify a unique neural mechanism or show an objective is universally bad.

## Intermediate observations, without checkpoint selection

| Condition | Step 31 clean / regressions | Step 93 clean / regressions | Final clean / regressions |
|---|---:|---:|---:|
| Task only |20/30 / 5|17/30 / 7|17/30 / 7|
| Preservation only |19/30 / 7|24/30 / 3|21/30 / 5|

The preservation-only step-93 total exceeds base accuracy, but still loses three
base-correct answers. It therefore fails the pre-existing preservation limit of
one regression, and cannot be selected as a successful repair. New gains and
old losses must be reported separately; shorter output alone is not a diagnostic
of correctness. All final conditions remain below the clean acceptance threshold.

## Integrity, review and scope

Two 186-step rank-one LoRA fits completed on Qwen3-4B-Instruct-2507. All 480 new
records downloaded without failures: 120 prior-adapter replay, 240 final answers
and 120 intermediate clean answers. Exact token-for-token replay passed before
training; exported adapters passed score equality and frozen-base restoration.
A CPU artifact verifier rechecked schedules, active/disabled losses, references,
outputs and finite exported parameters. This is not independent GPU replication.

There are 195 unique source-bound QA responses across the final and trajectory
packets. 167 identical responses inherit frozen earlier judgments; 28 new responses
were reviewed against sources with arm identities hidden, and decisions were
frozen before attribution. Review is by the assistant, not an independent human.
All 30 original documents remain. No post-result exclusion, new wording, selected
checkpoint or retuned hyperparameter is used. The base's earlier 23/30 clean
qualification failure remains a limitation. These are already-exposed development
inputs, so this diagnosis cannot establish generalization of a repair.

[Summary](summary.json) contains individual paired gains/losses and mean token
lengths. [Review freeze](review-freeze.json) binds the judgments; raw source text
and model responses remain private. [Protocol](../loss_isolation_protocol_v1.json)
fixes all identities and the two-hour limit.

## Cost and next decision

Provider execution took 2,349.99 seconds (39.17 minutes). Task-only fit time was
682.37 seconds including 60.41 seconds of intermediate generation; preservation-only
was 386.26 seconds including 60.49 seconds of intermediate generation. Each uses
186 training forwards and 2,064,384 trainable parameters, with 8,329,184 exported
bytes. Reference scoring and final evaluation are additional costs recorded in
[costs.json](costs.json). Fewer training terms do not mean equal training compute.

Do not change the neuron selector based on these results. The next design must
address the shared pressure to change useful behavior while retaining the learned
instruction distinction. Distribution retention and less restrictive task targets
are candidate controls, not established fixes. In particular, short hard-label
self-distillation is not equivalent to retaining the base output distribution.
A future repair needs validation separate from these exposed examples and matched
LoRA/random/guided comparisons before any superiority claim. No follow-up run is
launched by this report.
