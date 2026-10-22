# NeST source and CPU fidelity audit

The [official paper](https://arxiv.org/html/2602.16835v2) links this
[nine-file archive](https://anonymous.4open.science/r/nestSP27-1C20/).
Browser retrieval succeeded on22 October after direct HTTP requests failed.
The [manifest](source_manifest.json) pins the ZIP and every file by SHA256.
No license file was present; upstream source remains outside the repository.

`check.py` requires these pinned files and compiles only inspected tensor/parser
definitions. It does not import the training scripts, load pickles or models,
install the upstream environment, or run a trainer. Example after downloading
and unpacking the archive:

```sh
python -m experiments.nest_source.check --source /path/to/nest-source --output /tmp/nest-check.json
```

The [current-runtime results](cpu_checks.json) and
[PyTorch2.6 results](cpu_checks_torch26.json) cover:

- **Shared-row update fidelity.** Our compact grouped-row operation matches the
  actual `ClusteredRowParametrize` forward and input/update gradients in FP64
  and FP32. Merged weights are identical and unselected rows stay unchanged.
  This verifies one ingredient, not the full discovery or safety workflow.
- **Precision distinction.** A synthetic BF16 cancellation case distinguishes
  materializing a rounded modified weight from adding a compact output branch.
  A port must declare this numerical choice. This does not explain our completed
  Qwen live/merge failures, which already exist before merging.
- **Pooling scope.** The released activation hook takes a maximum over all
  positions without an attention mask. In a controlled fixture, a padding value
  wins that maximum. The effect on pretrained selection is not measured here.
- **Selection equation.** The code uses absolute standardized coefficients plus
  positive raw weights, whereas paper Eq.8 uses a positive standardized tail.
  A finite coefficient fixture demonstrates different selected indices. The
  fixture supplies NumPy's equivalent population-standard-deviation calculation
  in place of SciPy; no actual published probe coefficients were evaluated.
- **Judge predicate.** The released success predicate counts a parsed safe,
  non-refusing judge result just as it counts an unsafe, non-refusing one. It
  also scans the judge text for refusal phrases. The paper describes unsafe
  non-refusal evaluation. Controlled strings demonstrate the distinction;
  its frequency and effect on reported scores are unknown.

The source anchors are `1_get_safety_neuron.py:59` for pooling,
`2_sn_tune_cluster.py:103` for the parametrization, `:136` for selection,
`:262` for evaluation and `:464` for training arguments. The released default
is one epoch; the paper's SectionVI/TableI specifies five. Main training uses
up to10,000 harmful and10,000 benign examples, so this is not a nine-example
editing protocol or evidence that an arbitrary selector works with few labels.

These are reproducibility distinctions, not a replication of headline safety
results or a conclusion that those results are wrong. They also do not validate
our own selector. The useful implication is to preserve a precise source mode
when comparing with NeST, explicitly distinguish training precision, mask
padding in our proposed discovery workflow, and judge the desired behavior
rather than refusal alone. Cluster sharing is now verified at the actual core
operation; its benefit for our target tasks still requires controlled experiments.
