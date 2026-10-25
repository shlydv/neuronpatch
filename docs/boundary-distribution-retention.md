# Distribution-retention control

Status: optimizer integration tested; GPU worker not yet submitted.

The [loss-isolation result](../experiments/instruction_boundary/complete_loss_isolation_v1/REPORT.md)
shows that both short-answer task training and short-answer preservation training
can reproduce the same clean failures. The next control changes the preservation
objective while holding its token trajectories fixed. This avoids attributing a
simultaneous change in answer style to the KL objective.

## First comparison

Keep the 186 diverse SFT task examples, rank-one all-projection adapter, initial
seed, Adam schedule, learning rate and 31 clean examples unchanged. Replace clean
answer NLL with mean token KL(reference || edited), initially with weight 1.
Use the exact same clean prompt and source-reviewed answer tokens, including EOS,
and the same seeded clean-example order. This is a standard regularizer, not a
novel method. Weight 1 is a bounded control, not a claim of optimal tuning.

Cache base logits at those prediction positions before attaching any adapter.
The exact tokenizer/config calculation gives 119 prediction positions across
31 responses, vocabulary 151,936, FP16 cache 36,160,768 bytes; the longest response
cache is 3,038,720 bytes. A 64 MiB host-cache cap leaves explicit headroom without
quantization or truncation. Only one response reference moves to GPU per update.
Actual forward/backward peak memory still needs to be measured on the T4.

The existing `preservation_kl` primitive supplies the loss. The experiment-local
cache binds the base and tokenized trajectory identities. The shared optimizer
records `preserve_kl` separately from `preserve_nll`; the unused field is null.
Fifteen focused CPU tests pass, including direct Adam agreement for KL plus SFT,
unchanged default behavior, ablation lifecycle and half-precision worker checks.

## What this comparison can establish

It tests whether matching probabilities on the *same forced contexts* produces a
better training tradeoff than reinforcing the hard reference tokens. It does not
test complete base-response distillation: the base may normally produce different
prefixes. A subsequent base-generated-trajectory comparison must be separately
identified if needed, rather than quietly changing this control.

Evaluate complete generated answers, paired clean regressions, legitimate task
changes, attack resistance, lengths and costs. Retain every failure. The old
30-document set is exposed development data; no result on it establishes fresh
transfer or qualification. A promising repair must proceed to unseen inputs and
unrelated-task retention before comparing guided/random neuron supports with a
competent LoRA baseline. The original library advantage remains unproven.

Before submission: integrate cache preparation and hashing into the runner,
measure initial reference KL and finite gradients, replay the existing combined
adapter, and verify export/reload and evaluation in the actual tiny-model worker.
Do not add new neuron selection, target wording, checkpoint selection or multiple
rates to this objective control.
