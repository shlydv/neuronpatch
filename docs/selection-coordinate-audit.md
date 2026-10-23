# What our gradient score does and does not measure

23 October 2026. Analysis while the frozen relation-learning comparison runs;
no selector or GPU protocol change. This is a limitation of interpreting a raw
gradient score, not an explanation established for our earlier failures.

For a gated MLP, write its output as

\[
y=D[\operatorname{SiLU}(Gx)\odot Ux].
\]

Rescale one up-projection row by a nonzero constant \(c\), and the corresponding
down-projection column by \(1/c\). The MLP's function is unchanged. Under this
change of coordinates, the loss gradients for that neuron transform as

\[
g'_G=g_G,\qquad g'_U=g_U/c,\qquad g'_D=cg_D.
\]

Our paired squared-gradient score therefore becomes

\[
S'_j=\|g_{G,j}\|^2+c^{-2}\|g_{U,j}\|^2+c^2\|g_{D,j}\|^2.
\]

The same statement holds after averaging these squares across examples. Thus
raw empirical-Fisher group magnitude is not a function-invariant definition of
neuron importance. A scalar gate multiplying the complete intermediate feature
has invariant loss derivative under this specific rescaling, because the
feature-times-down-column contribution is unchanged.

A tiny float64 gated-MLP check, seed 23, rescales the initially lower-scoring
of two neurons by 1000 and compensates its outgoing column. Outputs change by
at most 5.56e-17, but the top-ranked neuron changes: scores go from
[0.0341, 8.5790] to [8090.9013, 8.5790]. Gate derivatives remain unchanged.
The [numeric record](../experiments/relation_learning/coordinate_check.json)
is an algebra demonstration, not an LLM benchmark or a proposed best score.

This does **not** make raw-gradient selection invalid for learning: fixed
coordinate-wise optimization also changes under reparameterization. A score can
be useful for a particular optimizer without being an intrinsic importance
measure. Nor does gate sensitivity establish that training incoming/outgoing
weights will generalize or preserve unrelated behavior. Deletion, small gate
changes, and many Adam updates are different interventions.

This distinction has relevant established precedent. [Molchanov et al., section
3](https://arxiv.org/html/1906.10771) define pruning importance through loss change
on removal, derive weight-times-gradient and gate-based approximations, and
distinguish the square of a group sum from a sum of squared contributions.
Their work evaluates pruning; it does not establish the best sparse adaptation
support for our task. Gate-based scoring would not be a novel idea.

Decision: finish the current comparison. If selection loses, investigate actual
updates and errors before replacing its score. Any future claim must specify
the intervention and optimizer it predicts, and beat the existing baselines;
calling a score "neuron importance" is insufficient.
