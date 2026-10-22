# Small weight changes are not sufficient protection

This retrospective audit reads all 24 existing saved patches and their complete
answer scores. It performs no fitting, model forwards or new test-set evaluation.
The [machine-readable record](update_geometry.json) includes every fitted state,
all input hashes and every smaller-damaged/larger-clean score witness. Reproduce
it with `python -m experiments.paired_mlp.update_audit --help`.

For development case1 at the low learning rate:

| Intervention | Nominal weight-update norm | Target answers /4 | Frozen scored preservation losses |
|---|---:|---:|---:|
| Gradient-selected paired neurons |2.138|4|4|
| Random paired neurons, seed11 |4.360|4|0|
| Rank-one MLP LoRA |2.269|4|1|

The [full answer review](REPORT.md) identifies two genuine errors in the guided
row: the injected country appears in an unrelated Lakers answer, and the oxygen
symbol changes from O to O₂. Its other two scored losses, and LoRA's single
scored loss, are valid formal UK names missing from the frozen aliases. Thus the
guided patch changes the weights less than either comparator while causing
genuine interference that is absent from both comparator rows on these probes.

This is not limited to alias-dependent comparisons. In case0 at the high rate,
the guided norm is7.420 versus7.797 for random seed11; the guided condition gets
only1/4 target answers and loses ten initially accepted preservation answers,
while that random condition gets4/4 with zero scored preservation losses.

Within the same case, support and parameterization, increasing the learning rate
increases the nominal norm in all12 matched pairs. Nine pairs have strictly more
scored preservation losses; three remain tied. Learning rate, trajectory, norm
and update direction change together, so this does not isolate a causal effect
of norm. Across supports there are12 same-case smaller-damaged/larger-clean
witnesses under the frozen scoring contract. These observations rule out perfect
monotone separation of these scores by one global norm threshold, not every
possible statistical use of update size.

The norm is the Frobenius norm of the nominal prequantization weight increment.
For sparse patches it uses the saved deltas; for LoRA it uses effective scaled
BA, not the norm of the factor matrices. The helper has a dense-product numerical
oracle test. This metric does not measure model sensitivity, relative changes to
base weights, FP16 merge rounding or output-distribution changes.

The useful implication is limited: smaller patches or earlier stopping alone
cannot be assumed to protect unrelated behavior. Stopping remains worth the
already-frozen checkpoint test. A subsequent candidate needs to measure the
functional effects of an update on correction and preservation examples under
the same procedure for guided, random and LoRA supports. This audit does not
validate a new selector or a safety rule. All outcomes were previously exposed,
there are only two development facts, and fresh confirmation remains unused.
