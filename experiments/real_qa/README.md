# Real-document evaluation preparation

Source: [official SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/),
CC BY-SA 4.0. Raw files are kept outside Git at `/private/tmp/neuronpatch-squad`;
URLs, SHA-256 hashes and structural counts are in `source_manifest.json`.
No model outputs have been generated for these data in this project.

Next protocol must split by article, retain all accepted answer aliases, and reject
or handle oversized full contexts explicitly. Never truncate away evidence while
retaining the original answerability label. Use normalized exact match and token
F1, separately measuring unsupported answers and unnecessary refusals. Preserve
source attribution; the library's MIT license does not relicense the dataset.

These public data may have appeared in model pretraining. New to our experiments
does not mean uncontaminated. Independent counterfactual evidence pairs and a second
task family remain necessary for a stronger generalization claim.

Unlike our synthetic records, the base may answer supported questions incorrectly.
Reference KL must not blindly preserve those wrong outputs: label reference-correct
training cases separately from correction cases, or explicitly compare that policy
with an all-supported reference constraint. Do not use final-test correctness to
construct training masks. This is a design requirement, not yet a tested recipe.

This directory prepares data provenance only. No real-document improvement claim
or experiment protocol has yet been committed.
