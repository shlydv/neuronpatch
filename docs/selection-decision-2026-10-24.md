# Selection must predict useful changes, not just recognize inputs

The authority comparison is already running under its frozen protocol. This note
sets interpretation boundaries before its outputs are read; it changes neither
the worker nor its success criteria.

## What the source methods actually support

[NeST, sections IV–V](https://arxiv.org/html/2602.16835v2) selects neurons from
harmful/benign activation probes and clusters their activation profiles. Its
case study also checks whether within-cluster weight gradients align more than
between-cluster gradients. That is empirical support for sharing updates in its
setting, not an identity between activation similarity and update compatibility.
Its training uses20k harmful/helpful examples; our32-document, single-template
training distribution is a materially different generalization problem.
The previously inspected source also differs from the paper in details recorded
in the [source audit](../experiments/nest_source/README.md).

[FiNE, section IV](https://opanhw.github.io/fine-unified-editing/static/pdfs/manuscript.pdf)
scores token contributions using activation times a projected output direction,
then optimizes selected output weights with an editing objective and auxiliary
losses. Its score is connected to a particular output token. A binary prompt
classifier is not an interchangeable implementation of that selector. Neither
score alone guarantees preservation or transfer of a trained behavior.

[Hase et al., NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)
find that causal-tracing localization does not predict the best MLP layer for
ordinary fact replacement in their setting; the edit layer itself is more
predictive. Their [official repository](https://github.com/google/belief-localization)
documents comparisons with constrained fine-tuning, ROME and MEMIT and several
editing objectives. This is evidence against assuming localization equals
editability, not a result that all neuron selection is useless. We inspected
the paper and repository documentation here, not a new reproduction of its code.

## Four separate questions for neuronpatch

1. **Recognition:** can the selected activations distinguish the intended
   conditions? Our old probes fit all training labels but did not establish
   held-out recognition, let alone transfer after training.
2. **Trainability:** can updates in this parameterization improve the objective?
   The old guided patch learned redirect examples, so it passed this limited test.
3. **Transfer:** does the trained change help new wording and documents? The old
   guided patch failed the claimed-authority family despite its larger initial
   reduction in same-template calibration loss.
4. **Preservation:** does it retain valid instructions and correct answers?
   Exact marker controls passed previously, but clean-answer regressions remained.
   Those controls do not establish broad capability retention.

The new matched prompts address a recognition shortcut: payload words no longer
separate the discovery classes. They do not directly optimize transfer or prove
trainability. Moving a payload across the JSON boundary still changes token
positions and length. Even a successful classifier can exploit those cues.
Moreover, changing discovery profiles also changes grouping; a positive result
would support the full discovery recipe, not isolate the coefficient ranking.

## How we will interpret the running comparison

| Observed result | Supported next conclusion |
|---|---|
| New patch beats old guided, both random seeds and approaches LoRA with preservation | Replicate the complete recipe across seeds and tasks; investigate which selected groups mediate the gain. No general superiority claim yet. |
| New patch improves old guided but remains worse than random or LoRA | Better contrast helped, but the selector has not justified its cost or a library default. |
| New patch still fails while random/LoRA transfer | Fixing this discovery contrast was insufficient. Do not launch another arbitrary threshold/pooling variation; investigate update compatibility or selection by actual training outcomes. |
| Every method fails the new grading wording | Shared training coverage or task difficulty becomes a stronger concern. This alone does not distinguish those explanations. |
| Injection improves but full-document legitimate commands fail | The intervention may be ignoring final instructions rather than learning authority. Such damage fails preservation. |

These conclusions require source-checked complete answers, paired documents,
raw regressions, the frozen three-hour limit and the saved comparator hashes.
The source-only audit is already fixed before inspecting outputs. If a run times
out or fails integrity checks, missing endpoints are not negative method results.

## Product implication

The proposed value remains a compact, deployable adaptation that users can trust
on held-out behavior. We have working compact patches, not evidence that our
selector reliably provides that advantage. A selector should therefore remain
experimental until it beats strong controls after accounting for discovery cost,
all attempted fits, artifact size and regressions. The completed comparison
already shows that fewer trainable scalars did not make training faster.

Choosing parameters by actual training outcomes is a research direction, not an
unclaimed invention or an automatic next GPU experiment. Existing attribution,
subspace and gradient-based methods in [the research record](research.md) must
remain controls. The active run should resolve the current targeted question
before another selection design is promoted.
