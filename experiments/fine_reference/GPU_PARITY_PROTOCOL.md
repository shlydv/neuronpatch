# Library port validation

This check follows the running upstream-only 16-case comparison. Do not cancel
or change that comparison to introduce library code.

Use the same pinned GPT-J checkpoint, dataset file, upstream source, single
CUDA0 placement, FP16 base/BF16 deltas, seed42, k5 per target token, 50-step cap,
objective coefficients and stopping threshold. Test cases0 and1 with both
contribution and random selection. They are exposed development examples and
are useful for port validation, not a fresh quality claim.

Load the original and ordinary Hugging Face models sequentially. Record baseline
and edited greedy generations on a fixed subset: rewrite, rephrase, first
subject alias, and the first probe of each locality category. Preserve selected
occurrence order; compare the learned deltas and sequentially merged columns.
Export/reload each library edit and verify its live generations. Restore the
base after every condition and verify all fixed baseline generations.

Success requires exact supports, learned deltas, merged columns, baseline and
merged generations, plus exact compact reload. A discrepancy is an investigation
result; preserve it, do not loosen tolerances after seeing it. Live-hook versus
merged generations are recorded separately because mixed-precision rounding can
differ. Exact agreement on a few prompts is not proof of universal equivalence.

The planned isolated comparison uses Python3.11, torch2.4.1 and Transformers4.40.0
for **both** implementations. Torch2.4.1 meets the library's core dependency.
Transformers4.40.0 is retained for the original modified model, below the optional
general-purpose integration extra's lower bound; this is an explicitly pinned
reference experiment, not the normal installation recipe. The first16 reference
used torch2.0.1 and remains a separate result. Do not silently combine outcomes
from these runtimes.

Before allocating GPU time, run the local complete source check in the isolated
Transformers runtime and the regular test suite in the current library runtime.
The former matches exact training updates; the latter includes an ordinary
Hugging Face GPT-J integration test without downloads. The GPU script and all
bundled library files must be hashed in submission provenance.

If the full-size port passes, select the next method change from the completed
16-case output audit. Main quality targets are generated rephrase transfer and
preservation of unrelated answers, with original-prompt success as a necessary
condition. Keep all exposed cases developmental; freeze the candidate and its
decision rules before testing new cases. A small numerical port check is not an
improvement result, and this protocol does not claim NeST or SafeNexus replication.
