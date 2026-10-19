# Data-size learning curve: preparation and baseline

Keep the pinned Qwen model, native prompt, extraction/UNKNOWN targets, aliases,
article partitions and 512-token full-context eligibility unchanged. The first
stage measures the base on expanded training, calibration and validation sets,
verifies original-prefix generation parity, and reports old and additional rows
separately. It does not evaluate confirmation.

The eventual broad-LoRA learning curve uses nested 128/512/2048 training prefixes.
Reference-correct preservation anchors must be selected only from the corresponding
training prefix. Their number/exposure is reported, because larger data also changes
preservation coverage. Keep learning rates, rank, eligible layers and objective fixed
across sizes, and report optimizer steps as well as epochs: equal epochs imply extra
compute at larger sizes. A fixed-step comparison is distinct from an equal-epoch one.

Do not use the added validation outcomes to change the data filters or relabel the
corpus. Errors in scoring may be investigated, but any revised scorer must apply to
all methods and remain separate from unchanged benchmark scores. Automatic exact
match is not semantic accuracy. The old validation rows are already development
data; added validation rows become development data when examined. Confirmation
stays reserved. This stage provides no claim of neuron-selection superiority.

First curve (recorded before scaled training): one epoch at each size, rates .0001
and .0003, all-MLP rank-2/alpha-2 LoRA, replay KL coefficient 1, two supervised rows
plus two anchor rows per step. Steps are 64/256/1024 at sizes 128/512/2048. Reuse the
existing one-epoch 128 checkpoints and evaluate them on the same 256 validation
rows as the larger fits, checking the original 32 outputs match. Hold back final
confirmation. Report the 224 added validation rows separately from the 32 already
used in development. One epoch bounds compute and avoids assuming the previously
observed multi-epoch validation deterioration is beneficial; it does not claim each
size has been optimally trained. Later fixed-step/epoch curves remain distinct.
