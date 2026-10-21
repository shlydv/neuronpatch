# Does restricting updates to selected neurons help subject binding?

**Status: running on Kaggle**, kernel 139988939 version 1. Accepted source matches
the frozen upload byte for byte. Dependency installation has completed and model
loading has started; no full-size result is available yet. See the
[launch record](launch_v1.json).

This is a prospective development comparison after the completed
[preservation pilot](../subject_binding/pilot_v1/REPORT.md). That pilot retained
6/6 requested answer starts on untrained wordings but still spread the replacement
to 4/8 unseen subjects. No candidate is currently a validated editing recipe.

The question here is whether the remaining interference comes from the narrow
support or from the correction/preservation objective shared by all updates.
This is not a fresh benchmark or a novel selection algorithm.

## Comparison fixed before inference

One bounded worker, two exposed facts, four conditions per fact, eight fits:

| Family | Learning rate | Trainable scalars per fact |
|---|---:|---:|
| Saved worst-prompt-gradient support | .001 | 20,480 / 81,920 |
| Mean-gradient support | .001 | 20,480 / 81,920 |
| Broad official PEFT LoRA, rank 2 | .0003 and .001 | 4,128,768 each |

Both sparse methods select five ordered occurrences per target token from the
same MLP down projections, excluding the final three layers. Consensus reuses
the exact saved occurrences; mean uses the norms of averaged gradients from
the same three correction prompts. Repeated columns retain independent rows.
The broad control covers attention q/k/v/out and MLP fc_in/fc_out in all 28
blocks, including the final three. It deliberately has much more capacity;
this is not a matched-parameter efficiency claim.

All new trainables are FP32; the base remains frozen FP16. Each starts from a
zero-output update and trains for exactly 50 Adam steps. The common objective is
the worst target-token/prompt NLL, mean prefix KL, the existing repetition term,
and weight 10 answer-distribution preservation on three other subjects. Preserve
the same first four base-generated prediction contexts, averaging positions
within each prompt and then prompts equally. No occurrence norm or LoRA factor
norm is used: those penalties would measure different things.

The earlier BF16/gamma .1 patch remains historical evidence, not a matched
condition. These changes are explicit in the new [protocol](protocol_v1.json).
No new correction examples, formatting variants or protected subjects are added.

## Selection and success

Choose the smallest offered learning rate in each family whose final **merged**
FP16 model passes both target NLL <= -log(.9) and every calibration-token
KL(base||edited) <= .1 in both cases. No qualifying rate means no selected
candidate. The selection function receives only training/calibration records.
All settings are reported; no extra rates are tried after seeing outcomes.

Count criteria for a selected candidate: 2/2 original answer starts, at least
5/6 untrained wordings, at most 1/8 off-subject replacement leaks, and zero
losses of baseline-reference answer starts in evaluation or unrelated utility.
Full capped continuations also require qualitative review. A prefix-count pass
alone never establishes usable answer quality.

All these cases and probe banks are already exposed development data. Even a
clean result requires confirmation on fresh edits, new subjects crossed with
new wordings, and stronger complete-answer evaluation. The prepared eight-fact
fresh bank remains unused by this worker.

## Integrity and execution

Before any new inference, replay exact saved base and edited reference token
sequences. Export/reload each LoRA artifact, require exact FP32 factors and live
calibration logits, then evaluate the actual merged FP16 model. A single CPU
snapshot restores all mergeable projections after LoRA; never subtract updates
to restore rounded FP16 weights. Check full base fingerprints after every fit.
The worker uses CUDA0 and has a 7,200-second runtime limit (7,500-second outer
worker timeout). No second GPU model copy is needed.

The tiny actual GPT-J worker test covers all eight fits, both representations,
export/reload, merging, restoration, calibration-only selection, independent
rescoring and 530 decoded output records. The common sparse objective is checked
against the existing trainer; LoRA accumulated gradients are checked against a
joint objective. Both installed modern PEFT and pinned PEFT 0.11.1/Transformers
4.40 are tested locally. Full-size GPU results remain pending.

```bash
python -m experiments.adapter_comparison.prepare \
  --binding-data artifacts/subject_binding_v1/frozen_ready \
  --output artifacts/adapter_comparison_v1/data
python -m experiments.adapter_comparison.build \
  --data artifacts/adapter_comparison_v1/data \
  --output artifacts/adapter_comparison_v1/kaggle_bundle
```

Use fresh output directories. The builder includes only source and frozen data;
no model weights, credentials or raw previous generated continuations.
