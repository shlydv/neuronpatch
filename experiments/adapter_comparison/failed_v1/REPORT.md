# Attempt 1: runtime dependency failure before LoRA

Kaggle kernel 139988939 version 1 terminates with return 1 after 1670.75 seconds,
about 27.8 minutes. It completes the two case-0 sparse fits, then fails at the
first LoRA attachment. No LoRA optimization starts; case 1 is not run. This is a
packaging failure, not evidence about the comparative methods or Kaggle queuing.

PEFT 0.11.1 imports BOFT while loading its tuner registry. That imports
`torch.utils.cpp_extension`, which imports `setuptools`. The isolated `uv venv`
contains the explicitly installed training dependencies but no setuptools.
Earlier developer environments already contained that package and masked the
omission. This preflight gap was our error.

The same missing-package failure is reproduced in a clean local Python3.11 /
Torch2.4.1 / PEFT0.11.1 environment. Explicit `setuptools==70.3.0` fixes import;
all 12 adapter/comparison tests then pass there, including the actual tiny GPT-J
eight-fit worker, adapter round trips, merging, exact restoration and independent
rescoring. The corrected worker imports the adapter stack before downloading the
model. macOS CPU testing does not reproduce CUDA kernels or the full-size fit.

The accepted source was independently pulled byte for byte before this failure.
All 22 downloaded artifacts remain local. The saved eight cold sequences,
two merged calibration records, 50-step budgets and FP32 scalar counts are
independently checked. Complete generated evaluation outputs are preserved but
have not been inspected to tune the retry. No global candidate can be selected
from this partial run. Do not report its training success as comparative success.

The corrected retry changes only the explicit packaging dependency and upfront
import validation. Training data, loss, neuron supports, LoRA rank, offered
learning rates, seed, 50 updates and success/selection rules are unchanged. It
reruns the registered eight fits from zero; the partial version is not spliced
into the completed comparison. Version 1 is confirmed terminal before submission.

See [protocol](protocol_v1.json), [accepted launch](launch_v1.json),
[source manifest](submission_v1.json), [provider](provider.json),
[execution log](execution.log), [partial fits](fits.json) and
[integrity checks](integrity.json). Raw generated text/tokens remain private.

After the corrected retry was frozen and accepted, the recovered case-0 outputs
were decoded and scored as [explicit partial evidence](PARTIAL_AFTER_RETRY_FREEZE.md).
This later inspection does not tune the retry or turn the failed run into a
complete comparison.
