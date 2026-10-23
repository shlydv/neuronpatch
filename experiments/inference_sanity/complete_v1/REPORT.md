# FP32 and device placement do not fix these observed failures

All **12 generated token sequences are identical** across single-GPU FP16,
two-GPU FP16 and two-GPU FP32. The eight original input replays match every
saved baseline token and the entire first-token logit vector. The two user-only
prompt contrasts do not repair the tested factual errors. This result rules out
these precision/device settings as an explanation of the observed output
failures; it does not certify every aspect of inference or general model ability.

The [bounded Kaggle diagnostic](https://www.kaggle.com/code/shlydv/neuronpatch-inference-sanity-v1-20261023)
completed version 1 with exit 0: 36 generations, zero training, 250 seconds of
worker computation and 392 seconds including provider setup. All modes match the
original model's weight fingerprint after canonical FP16 conversion. Saved
configurations and the frozen source establish the actual requested dtypes and
placement; no CPU model inference was used.

[Independent checks](integrity.json) verify the frozen inputs/protocol, all
36 decoded outputs, stopping flags and first-token argmax agreement. Run
`python -m experiments.inference_sanity.analyze` with the private artifacts and
cached tokenizer to reconstruct them. First-token vectors are exactly equal
between the two FP16 placements. FP32 changes numerical logits (largest absolute
difference about 0.197), but not any first-token argmax or generated answer.
The largest measured FP16-to-FP32 first-token KL is 0.000391 nats. This is a
small-input result, not an assertion that precision never affects outputs.

## Full-output review

Every saved response was read. The same 12 texts occur in each mode:

| Input | Observed outcome in all three modes |
|---|---|
| Ragnarok date, original instruction | Wrong year: 2018 rather than the archive's 2022 |
| USA 2022 World Cup group | Wrong Group D; longer continuation does not repair it |
| Argentina 2022 World Cup group | Wrong Group D |
| Pixel Watch start of sales | Wrong March date relative to the archive's October date |
| Spanish Super Cup winner | Correct Real Madrid |
| Alcaraz semifinal, supported evidence | Wrong Sinner despite explicit Tiafoe evidence |
| Alcaraz semifinal, missing evidence | Wrong Sinner instead of UNKNOWN |
| CAA venue, supported evidence | Supported city Los Angeles; less specific than the reference venue |
| Ragnarok date, user-only | Same wrong year |
| Argentina group, user-only | Contradictory participation claims, then wrong Group D; cut off at 96 tokens |
| 17 multiplied by 23 | Wrong 401; arithmetic result is 391 |
| Capital of France | Correct Paris |

Eleven of twelve responses terminate naturally in each mode. The remaining
user-only Argentina response is incomplete in all three. The model can follow
some instructions and use supplied evidence; these examples do not support a
claim of total inference failure. They also do not justify treating its
knowledge-only answers as a strong general-capability baseline.

## Decision

Close the precision hypothesis for these examples. Do not keep changing dtypes,
prompt wording or decoding to fish for a favorable answer. The earlier retrieval
qualification remains failed: it does not supply enough repeated evidence-use
errors to warrant fitting adapters. The semantic event-relation mistake is real
in the observed outputs, but one case cannot validate a reusable training recipe.

Next method work needs an independently varied, well-labeled failure family and
an equally supervised LoRA baseline before claiming a sparse selection benefit.
Neither this negative diagnostic nor the completed retrieval test moves us to
paper superiority. Both prevent spending more training compute on an unsupported
explanation.
