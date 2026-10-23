# Inference sanity check before further adaptation

The [retrieval baseline](../retrieval_evidence/complete_v1/REPORT.md) had near-ceiling
missing-evidence performance but unexpectedly poor question-only answers.
Before assuming those outputs describe normal checkpoint capability, check
precision, device placement and two system-prompt contrasts. No inference bug
has been established.

The [frozen protocol](protocol_v1.json) uses twelve prompts and three modes:
original single-GPU FP16, split-GPU FP16, and the same split-GPU map in FP32.
Eight prompts replay observed development inputs, two remove the system prompt
from question-only inputs, and two test arithmetic/capital knowledge. Every
mode receives identical prompts and greedy decoding capped at 96 tokens.
The longer cap is explicitly a new diagnostic, not a replacement score for the
48-token baseline. Canonical FP16 weight fingerprints must match its base.

Interpret the original-mode replay first, then the same-device-map precision
contrast. A mismatch between one- and two-GPU FP16 cannot be attributed solely
to precision. Prompt controls and full continuations distinguish other possible
causes. Neither a few improved outputs nor matching outputs establishes broad
capability or proves precision always harmless.

The source checkpoint declares BF16 in its
[pinned config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json).
The [official usage](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
uses its chat template and describes a non-thinking model. The diagnostic keeps
that template and uses FP32 as a reference on T4 GPUs; it does not assume BF16
native acceleration is available there. No source weights run on the laptop.

One bounded private Kaggle job: 2,100-second worker timeout, 3,600-second provider
limit, no paid scaling. Run failure is not permission to restart automatically.
Raw historical documents stay private under the source dataset license.

The [completed result](complete_v1/REPORT.md) finds identical generated tokens
in all three modes. This diagnostic is closed; it did not identify a precision
fix or justify adaptation.
