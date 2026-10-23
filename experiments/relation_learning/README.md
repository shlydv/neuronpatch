# Relation learning pilot

Status: version 1 stopped with a nonfinite gradient during the second LoRA
setting after one completed fit. See `failure_v1.json`. Accepted source exactly
matches `frozen_v1.json`. The comparison is incomplete; no efficacy claim.
Held-out generations have not been reviewed and no rate has been selected.

Version 2 uses `protocol_v2.json` with the exact same input bytes, eight
conditions, rates and step counts. The only training change is optional
PyTorch GradScaler backoff: start at 128, halve on a nonfinite gradient, and
retry the same two examples without an Adam update. Stop if scale would go
below 1/128. Nonfinite forward losses still fail immediately. There is no
automatic scale growth or skipped training example. Every retry is logged and
independently checked. Tests induce real FP16 overflow and confirm identical
weights to a fixed-working-scale reference, plus no updates on unrecoverable
failure. This validates retry mechanics, not the cause of the GPU failure.
The completed first fit must be compared with v1 before reviewing efficacy.
Use `python -m experiments.relation_learning.replay --before V1_RESULTS
--after V2_RESULTS --output REPLAY.json`. It checks complete generated records
without displaying their content, adapter and teacher tensors, discovery scores,
metrics and shared history fields. A mismatch is reported, not hidden behind a
matching rounded accuracy. This replay check is separate from selecting rates.

The preceding relation-contrast probe found repeated errors distinguishing an
interview's qualification assessor from its final assessor, and a release's
announcer from its releaser. This pilot tests learning those distinctions, not
memorizing replacement facts. It does not claim to reproduce FiNE, NeST or
SafeNexus, or introduce empirical Fisher selection as a new algorithm.

## Frozen comparison

`data_v1/protocol.json` specifies eight fits of Qwen3-4B-Instruct-2507:
rank-one LoRA, layer-matched random paired neurons with two selection seeds,
and paired neurons ranked by mean per-example squared gradient norm. Each
method has learning rates 0.0003 and 0.001. Scalar budgets are approximately
matched; measured counts and discovery costs must accompany results.

All fits use the same 64 task examples and 20 preservation examples, with
64 optimizer steps. Fisher discovery uses eight specified training examples.
Training, validation and test names are disjoint. Half the 64 task-test rows
also change wording. The original 96 probe rows are exposed regression data,
never fresh evidence.

Run `python -m experiments.relation_learning.select --fits RESULTS/fits.json
--protocol experiments/relation_learning/data_v1/protocol.json --output CHOICES.json`
before inspecting test predictions. This chooses each method/seed's rate by
validation completion loss only, with lower rate breaking ties. Report both
rates as well as the chosen rate. No early checkpoint selection is performed.

After selection, use `python -m experiments.relation_learning.review --results
RESULTS --choices CHOICES.json --output REVIEW` to generate the complete review
queue and provisional exact-match transitions for all rates. This verifies the
saved selections against validation loss. It does not adjudicate semantics.
Every row starts with a null semantic review; inspect full responses, not just
exact misses. Separate correct supported answers, justified abstentions,
incorrect abstentions, unsupported names, other incorrect answers and truncated
responses. Count corrected and newly damaged items against the same base input.

## What would justify continuing?

A selected LoRA improving at least 8 of 64 task-test answers, while losing at
most two correct solved-family test controls, establishes learning headroom.
If Fisher improves the correction/preservation tradeoff versus both random
selections and LoRA, that supports a subsequent independent replication—not
a general superiority claim. Review full outputs, truncation and unsupported
answers; exact-string scoring alone is insufficient.

The eight unrelated utility questions are supplemental. Count retention only
where the base answer is correct, and report that denominator. They cannot
establish broad preservation. All conditions share templates, so 64 questions
are not 64 independent demonstrations. No automatic library-default change.

## Execution boundaries

Reuse the tested `experiments.paired_mlp.run` worker. Internal worker budget:
6300 seconds; process budget: 6600 seconds; planned provider budget: 7200
seconds. One free private Kaggle job, no automatic retry. An observer timeout
does not mean the GPU job stopped. Verify authoritative status before any
subsequent submission. Save source/data hashes and accepted kernel source.

The streaming selection accumulator avoids retaining gradient factors for
every discovery example. The tiny-model test independently checks its math
and executes Fisher, random and LoRA training, export/reload, merge and base
restoration. These checks establish implementation integrity, not efficacy.

## Completed result

See [the verified report](complete_v2/REPORT.md). All eight fits completed.
The learning criterion passes; a Fisher-selection advantage does not. Validation
choices, all predictions, semantic review, integrity checks and exact v1 replay
are saved in `complete_v2/`.

## Eight-example comparison

`low_data.py` freezes `low_data_v1/`: eight task labels plus the same twenty
preservation labels, 64 successful updates, the same parameter budgets and two
learning rates for LoRA, Fisher and two random supports. Fisher uses exactly the
eight task examples for discovery. This tests task-label efficiency, not an
eight-label total training budget or lower compute cost.

The 128 task-test questions use four fresh name sets and the existing two wording
variants; sixteen scope controls use a fifth fresh set. Eight fresh utility
questions supplement the earlier eight now explicitly exposed controls. The
32 validation questions remain the rate-selection bank. Save choices before
inspecting test predictions. This is new-name confirmation within a known
synthetic setup, not a new domain or a paired comparison with old test counts.

A promising selection result requires at least eight more correct task answers
than each selected comparator, gains in both wording variants and no additional
base-correct scope/utility regressions. It would trigger replication, not a
superiority claim. If LoRA and both random supports each reach at least 124/128,
retire this task as a selector discriminator at this parameter/update budget.
Otherwise review errors before deciding what follows. Report all rates and
costs. No selector, formatting or loss changes are included.

Pinned-tokenizer preflight passes all 336 rows (maximum 172 tokens); the original
64-example data hash is unchanged. Local tests cover split isolation, score
arithmetic and the actual tiny-model Fisher/random/LoRA training worker.
