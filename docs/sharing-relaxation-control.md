# Is fixed sharing the unnecessary constraint?

This is a possible controlled comparison after the pending supervision result,
not a new library method or a submitted GPU experiment.

Our grouped patch on selected rows is `Delta = G D`: `G` is a fixed one-hot
membership matrix, and `D` contains two trainable row updates. Replace this by
`Delta = B A`, where both factors are trainable, initially `B = G` and `A = 0`.
Keep the same selected rows and scatter their updates into the same projections.
Every grouped update remains representable by setting `B = G, A = D`; the
relaxed family additionally allows different signed mixtures for each row.

This is ordinary masked low-rank adaptation with a group-based initialization,
not a novelty claim. [SafeNexus equation 7](https://arxiv.org/html/2607.28969v1)
already describes masked LoRA on gate/up projections. On 24 October, its
[author-page](https://muzishen.github.io/) Code link still resolved to
[an unrelated model-merging repository](https://github.com/chenhangcuisg-code/Cross-Architecture-Merging-for-Large-Language-Models).
Thus this is a mathematical comparison, not verified SafeNexus code parity.

For our 72 projections, input width 2560, 32 selected rows, and two groups:

- Fixed sharing: `72 * 2 * 2560 = 368,640` trainable scalars.
- Learned mixing: `72 * (2 * 2560 + 32 * 2) = 373,248` scalars.
- Additional budget: 4,608 scalars, **1.25%**. This does not establish equal
  runtime, memory, optimizer behavior, or effective capacity.

The [CPU fixture](../experiments/instruction_boundary/sharing_relaxation_check.py)
checks arbitrary-delta representability and the actual library grouped forward.
At zero initialization, both have identical basis gradients; the mixing
gradient is zero. Their first Adam updates and outputs match exactly in the
float64 fixture. On the next step, mixing obtains a nonzero gradient and changes.
The fixture also exhibits different signed updates for two formerly tied rows.
This does not verify mixed-precision LLM training, export, or answer quality.

## When this comparison is useful

If repaired supervision works for LoRA but the grouped patch still loses useful
behavior, compare fixed versus learned mixing on the **same** support, with
fresh deltas, identical initialization of the shared basis, data, sampling,
optimizer, and steps. Do this for guided and random supports: changing support
and sharing simultaneously would again obscure the cause. Broad LoRA remains
a practical comparator, while this control isolates a narrower design choice.
The initialization removes an avoidable first-step discrepancy; subsequent
optimization differences are intrinsic to the parameterization being tested.

If every representation still fails the repaired supervision, prioritize the
objective and data diagnosis. If fixed sharing already matches LoRA with good
preservation, prioritize replication and unrelated-task checks. Do not launch
this comparison merely because it is easy to implement. Rank-two masked LoRA
still cannot copy dense changes outside its selected rows, and success in this
control would not prove that neuron discovery adds value.
