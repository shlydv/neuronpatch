# Response averaging dilutes the largest preservation changes

Read-only diagnosis completed for all 115 training anchors in all four saved
control/coverage fits. Both model distributions are teacher-forced on the same
gold answer and EOS, using the training masks and response-logit helper. Every
batch's recomputed mean KL matches the library preservation loss. These are final
checkpoint measurements, not historical per-step losses.

Among examples refused under the original decoder, the median first-token/mean-KL
ratio ranges from 4.0 to 6.1 across the eight fit/group combinations. The first
token has the largest KL in 135/139 refused example/checkpoint observations.
These observations overlap across fits; they are not 139 independent examples.

For the expanded higher-rate fit, the Windows-installer answer has 17 supervised
tokens: first-token KL is 1.505 nats but response mean is .108. In the exact-anchor
control it is 3.370 versus .214. That example is an added anchor, so the control
does not replay it; compare objective aggregation, not equal protection exposure.
The existing loss gives an isolated first-token discrepancy roughly 1/N weight
for an N-token answer. Teacher-forcing subsequent gold tokens can therefore yield
small later divergences even when generation would instead begin with a refusal.

This supports an aggregation hypothesis but does not prove a causal cure. The
decoder audit separately shows that repetition penalties can flip choices, and
larger average KL could also help. A fair follow-up needs both a worst-token
penalty and a stronger mean-penalty control; otherwise improvement could merely
reflect increasing regularization. Do not infer that activation selection or
clustering has failed from this objective-level limitation.

See token_kl_diagnostic.json for every token, and first_token_diagnostic.json for
raw prompt-boundary distributions and original generated outcomes. Neither is a
semantic accuracy metric. The next comparison will use an explicit common decoder.
