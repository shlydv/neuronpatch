# The measured guard stalls learning and confuses wording with damage

**Complete, independently verified, frozen advancement criteria failed.**
Eight guarded fits on Qwen3-4B produce806 generated observations. Risk-aware
selection, random selection and official LoRA each transfer3/8 requested answers;
gradient magnitude transfers1/8. All controls fail the complete-edit prerequisite,
so there is no informative superiority result. These are two exposed
counterfactual development edits, not corrections of real-world facts.

## Frozen comparison and complete answers

The [protocol](../protocol_v1.json) uses the same cached supports, approximately
1.33million trainable scalars, learning rate0.0003, correction examples and six
canonical preservation responses per case. Every method receives correction CE
plus base-anchored KL, up to60 Adam proposals, four backtrack scales and a
0.9 minimum training-token probability stop including EOS. A proposed update
must improve task NLL and retain mean/worst-token KL budgets0.02/0.2. Rejection
restores parameters, gradients, optimizer state, learning rates and RNG.

| Method | Case0 transfer /4 | Case1 transfer /4 | Accepted updates, cases0/1 | Stop, cases0/1 |
|---|---:|---:|---:|---|
| Gradient magnitude |1|0|4 /2|Blocked /blocked|
| Gain/risk selection |3|0|3 /2|Blocked /blocked|
| Random seed11 |3|0|10 /8|Training confidence /blocked|
| Official rank-one MLP LoRA |3|0|11 /8|Training confidence /blocked|

Every method retains the old US citizenship answer on the natural question
asking which country Leonardo DiCaprio holds citizenship in. Case1 never
produces the requested Wassoulou Empire on these transfer questions. The
successful training-confidence stops therefore do not establish transfer.

No scored loss occurs among49 initially accepted preservation observations
per method across both cases. Assistant review of all55 distinct
prompt/reference/output contexts, covering all806 observations, identifies no
new unambiguous preservation factual error in this bank. This small-bank result
is not general safety. Case1 risk-aware and random outputs each truncate on one
transfer question; live/reloaded/merged repetitions give six truncated
observations. No unexpected special tokens occur. See [analysis](analysis.json)
and the separate [completed review](review.json); the automated analysis's
review flag is not an automatic substitute for that review record.

All248 live/reloaded generated pairs match exactly. Live/merged generations
match247/248; the sole difference is `week` becoming `week.` for case1 gradient
selection. It changes no scored outcome. All eight reload teacher controls are
exact, and the independently checked final parameters, logits and recorded
acceptance decisions satisfy [integrity checks](integrity.json).

## The guard's contract is a substantive problem

The [reproducible saved-endpoint audit](../contract_audit.py) applies the same
frozen guard to the older50-step LoRA/random controls. All four previously
successful8/8-transfer controls fail it:

| Historical endpoint | Case0 mean /worst KL | Case1 mean /worst KL |
|---|---:|---:|
| LoRA |1.8868 /18.8124|1.6315 /19.3594|
| Random seed11 |1.8256 /15.1106|1.2960 /16.4856|

At the prefix `United States`, the base prefers EOS and edited endpoints prefer
` of`, completing another valid alias. For Huma Abedin in case0 LoRA, worst-token
KL is16.1402 while probability mass on the two covered valid aliases increases
from0.9992003 to0.9999521. For George Santos it increases from0.9998486 to
0.9999825 despite worst-token KL18.8124. Both responses remain accepted.

The audit computes an exact probability for the subset of complete EOS-ended
paths whose prediction prefixes were actually saved. It reports missing aliases
and never invents their conditionals. These are lower bounds on all acceptable
responses, not a semantic-equivalence oracle. Some auxiliary references also
disagree with the base output: David Goodwillie starts outside the supplied UK
set; the Travis Alexander row starts outside its supplied US set. Treating these
as simple retention examples conflates correction with preservation. This audit
does not adjudicate ambiguous labels or change the old scores.

All24 final rejected proposals in the six blocked fits lower task NLL, but each
violates worst-token KL0.2. Thus the observed stalls are preservation vetoes,
not absence of task descent among those attempted proposals. Rejected logits
were not saved, so their precise semantic effects cannot be reconstructed.

The new [response-set primitive](../response_sets.py) computes finite-set
marginal likelihood with explicit EOS, shared prefixes, duplicate elimination,
missing-context rejection and autograd. Independent probability trees distinguish
harmless alias redistribution from loss of accepted-answer mass. This is
standard probability marginalization, not a new algorithm or evidence of an
improved pretrained method.

Crucially, the historical teacher files do not contain logits for the unrelated
Lakers and oxygen questions where the selected patch failed. Complete generated
answers establish those errors, but the saved data cannot prove a probability
guard would detect or prevent them. See [all six audited endpoints](contract_audit.json).
Qualifying a replacement guard on both good and bad controls is the next step;
loosening this run's threshold after its failure is not.

## Cost and source provenance

The accepted [Kaggle version1](https://www.kaggle.com/code/shlydv/neuronpatch-guarded-training-v1-20261022)
returns0 in2234.71seconds,37.25minutes including setup and loading. The old
observer handles disappeared; the existing completed job was downloaded without
starting another GPU run. Provider session361183228 used the private T4x2
allocation, with the worker on CUDA0. The requested5400-second provider limit
was not exercised, so its enforcement is not established by this run.

All47 bundled files match frozen Git revision
`7db5edbff605b3a474a00d79b6d519a56db1520d`, all203117 raw script bytes match the
accepted downloaded source, and the executed submission matches the bundle.
Source SHA256 is
`bdd26afacde4c6557fb2df4ea9a9109730123583e8f9b4e0dcc92a3b65d0a6a4`.
See [source integrity](source_integrity.json), [provider](provider.json) and
[launch](../launch_v1.json). Later analysis/contract-audit code is postprocessing,
not part of the frozen GPU intervention.

Cached risk-aware attachment+fit+export totals137.43seconds across both cases,
versus64.62 for LoRA, about2.13times as long. Sparse fingerprinting still
dominates attachment/export. Cached discovery is excluded, so this comparison
cannot establish whole-pipeline speed.

Keep the frozen failure and the earlier genuine interference. Do not promote
this guard, selector or response-set helper to a recommended library recipe.
The [current method decision](../../../docs/current-method.md) separates the
contract repair, missing negative-control measurements and next method hypothesis.
