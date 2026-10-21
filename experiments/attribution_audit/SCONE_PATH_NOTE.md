# A path-gradient mismatch in the inspected SCoNE implementation

This CPU check finds a reproducible discrepancy between the stated integrated-
gradients formula and one inspected implementation path. It does **not** show
that the paper's benchmark results are wrong or that neuronpatch performs better.
No full LLM experiment or final-neuron-selection comparison has run for this audit.

The [paper, equation 1](https://arxiv.org/html/2609.00689v1#S2) differentiates answer
probability with respect to the interpolated activation. The paper-linked code
is inspected at commit `3b02f40eefa8e878db7c5f88849365b25359c5ad`:

- [Interpolation construction](https://github.com/HYU-ARK-Lab/SCoNE/blob/3b02f40eefa8e878db7c5f88849365b25359c5ad/models/generators/generator.py#L57).
- [Gradient target](https://github.com/HYU-ARK-Lab/SCoNE/blob/3b02f40eefa8e878db7c5f88849365b25359c5ad/models/generators/generator.py#L198).

The source SHA256 is
`2b4e29ccbd3d0ce3a163b54379062e0495b82e288f910125a736d074cf10972c`.
The audit extracts only the two inspected methods in memory, applies them to an
independent two-feature probability model, and changes only the gradient target
for the comparison. No upstream implementation is vendored or imported wholesale.

## Why the two gradients differ

For fixed endpoints, define

\[
v_\alpha=v_0+\alpha(v_1-v_0),\qquad
\operatorname{IG}=(v_1-v_0)\odot\int_0^1\nabla_{v_\alpha}P(v_\alpha)\,d\alpha.
\]

The inspected code differentiates through interpolation with respect to the
endpoint `w_ffn_activations`, instead of the intervened `batch_activations`.
The chain rule gives

\[
\nabla_{v_1}P(v_\alpha)=\alpha\nabla_{v_\alpha}P(v_\alpha).
\]

Thus its numerical sum contains an additional alpha weighting. With the actual
20-point grid, alpha is 0, .05, ..., .95. A constant path gradient is multiplied
by mean(alpha)=.475. This differs from ordinary left-Riemann quadrature error:
the correct rule is exact for a constant gradient.

## Results from executing the inspected methods

The independent model has valid answer probability

\[
P(x)=.4+.1(1.2x_0+x_1^p),\qquad x:(0,0)\rightarrow(1,1),\quad p\in\{1,3\}.
\]

Both cases have exact integrated gradients (.12,.10) and endpoint probability
change .22. Closed-form derivatives evaluated at the same 20 points provide an
independent finite-grid reference.

| Probability model | Inspected source attribution | Interpolation-gradient attribution | Same-grid reference | Top feature changes? |
|---|---|---|---|---|
| Linear, p=1 | (.057,.0475) | (.12,.10) | (.12,.10) | No |
| Cubic second feature, p=3 | (.057,.0676875) | (.12,.092625) | (.12,.092625) | Yes, 1 to 0 |

In the linear case, the source sums to .1045 instead of .22; changing the gradient
target restores completeness exactly. In the cubic case, the corrected finite
sum still has normal quadrature error relative to exact IG, but matches the
independent same-grid derivative reference. The extra endpoint weighting reverses
the feature ranking in this constructed example. Both Torch2.14.1 and clean
Torch2.4.1 runs reproduce these numbers in CPU float64.

## Implications and limits

This demonstrates a code/formula mismatch and a possible local ranking change.
It does not quantify changes to SCoNE's aggregated frequency selection, its final
five neurons, or real-model answer quality. Tokenization, baseline construction
and quadrature conventions also require separate validation before a reproduction.
The inspected repository defaults differ from its documented reported settings;
those overrides would need to be pinned as well.

Our current GPT-J comparison uses direct weight gradients, not this attribution
routine, so it is unaffected. This finding motivates analytical checks for any
future path-attribution selector; it is not an improvement claim for our method.
No issue, PR or message has been sent to the upstream authors.

## Reproduce without downloading model weights

Obtain the paper-linked repository at the pinned commit, then run from neuronpatch:

```bash
python -m experiments.attribution_audit.scone_path \
  --source /path/to/SCoNE/models/generators/generator.py \
  --output /tmp/scone-path-result.json
```

The script requires an exact source hash before executing its two selected
methods. Internal assertions compare against independent derivatives and the
linear completeness identity. See [diagnostic](scone_path.py),
[Torch2.14 result](scone_path_v1.json) and [Torch2.4 result](scone_path_torch24_v1.json).
