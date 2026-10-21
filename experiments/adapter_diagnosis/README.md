# Which learned update groups make the LoRA edit selective?

**Status: CPU-verified preparation; no full-size diagnosis result yet.**
The completed [parent comparison](../adapter_comparison/pilot_v2/REPORT.md)
favors broad low-rate LoRA over our outgoing-neuron patches. This diagnostic
reuses those exact calibration-selected factors; it does not refit adapters,
try new rates, change formatting or consume the eight-fact fresh bank.

The question is whether the broader baseline needs attention, incoming MLP
updates, outgoing MLP updates or the final three layers to retain correction
without spreading the replacement to other subjects. Finding a required group
would be conditional on this fitted adapter; it would not locate a unique
concept, prove how training formed it, or establish a better sparse method.

## Fixed interventions and controls

For each of the two exposed edits, evaluate the unchanged base and:

- Full selected LoRA.
- Only attention / without attention.
- Only incoming MLP / without incoming MLP.
- Only outgoing MLP / without outgoing MLP.
- Only final three layers / without final three layers.

The three projection families partition all adapted projections. Final-layer
coverage overlaps these groups and is an independent control. Partial adapters
are formed by zeroing excluded B factors, leaving retained A/B factors and base
weights unchanged, then using the same actual FP16 merge as the parent run.
This removes learned parameter updates, not base heads or activations. Apply
the same intervention at every token; there is no input-dependent mask or
renormalization. Different retained groups have unequal capacities and norms.

Require **all** base/full generations and calibration tensors for **both** edits
to replay exactly before any ablation. The saved parent files are independently
hash-bound to the verified completed comparison. Any mismatch stops the run.
Restore actual base weight snapshots after every merge; never subtract rounded
updates. No second GPU model is allocated.

Reuse all 53 prompts per case: six target wordings, three calibration, four
evaluation, eight previously exposed scope and 32 utility prompts. The ten
conditions per edit produce 1060 observations, including 212 replay controls
and 848 partial-adapter observations. There are 18 merged interventions,
**zero new fits**, and a3600-second worker bound (3900-second outer timeout).

## Predictions and useful outcomes

The hypothesis is that outgoing-only edits may learn a broad replacement,
while other projections help make its use conditional. This is an unconfirmed
explanation, not an interpretation to force onto the outputs.

Report every condition. A sufficient subset must satisfy both cases' target
NLL<=-log(.9) and all calibration-token KL<=.1, retain2/2 original and>=5/6
untrained target starts, leak on<=1/8 evaluation subjects and lose no initially
reference-correct evaluation/utility starts. A count pass is only conditional
development evidence and still requires all capped continuations to be reviewed.
No new candidate or learning rate is selected in this diagnostic.

If outgoing-only retains selective editing, that weakens our proposed missing
incoming-transform explanation and points toward support coverage rather than
requiring incoming updates. If other groups matter, a targeted parameterization
comparison becomes justified. A removal that breaks correction is not evidence
that the removed group specifically controlled preservation; report both.
Jointly trained groups can compensate for one another. If only the full adapter
works, do not invent a clean neuron story from the result.

Zero replacement leakage is not preserved general ability. The parent base is
weak, most utility prompts are not initially reference-prefix-correct, all
continuations are capped at40tokens, and later errors can follow correct starts.
Keep complete observed continuations and also inspect literal target mentions
beyond the start. No quality, novelty or paper-superiority claim is planned.

## Local checks and execution

The actual tiny GPT-J/PEFT test checks every retained merged matrix against the
unchanged full merge (and excluded matrices against base), complete two-edit
replay/intervention/restoration, independent scoring
of all saved tensors, and rejection of a changed control or reported metric.
Those checks run in the pinned Torch2.4.1/PEFT0.11.1 runtime without model downloads.
They establish implementation behavior, not a full-size mechanism result.

The pinned PEFT0.11.1 merge adds FP32 B@A before returning to base dtype.
Newer safe merging can round the update to FP16 first. The fixture verifies
both possible official rounding orders and exact within-runtime group removal;
it does not claim cross-version bitwise parity. The full-size diagnostic uses
the parent's pinned runtime and exact replay requirements.

```bash
python -m experiments.adapter_diagnosis.prepare \
  --parent artifacts/adapter_comparison_v2/kaggle_download/comparison-results/results \
  --integrity artifacts/adapter_comparison_v2/integrity_v1.json \
  --output artifacts/adapter_diagnosis_v1/data
python -m experiments.adapter_diagnosis.build \
  --data artifacts/adapter_diagnosis_v1/data \
  --output artifacts/adapter_diagnosis_v1/kaggle_bundle
```

Kaggle attaches the parent kernel's outputs as an input. The worker requires
their exact hashes, so a newer or incorrect attached version cannot silently
replace the selected factors. No model weights or credentials are bundled.
The frozen launch/source identity must be recorded before inference. After
completion, verify control replay, factor-group accounting, decoded generations
and independently reconstructed calibration before interpreting the outputs:

```bash
python -m experiments.adapter_diagnosis.verify \
  --results artifacts/adapter_diagnosis_v1/kaggle_download/diagnosis-results/results \
  --parent artifacts/adapter_comparison_v2/kaggle_download/comparison-results/results \
  --protocol experiments/adapter_diagnosis/protocol_v1.json \
  --decode --output artifacts/adapter_diagnosis_v1/integrity_v1.json
```

The overall goal remains a useful adaptation library with a demonstrated
advantage, not just an explanation of two counterfactual edits. Fresh crossed
subject/wording evaluation and a capable complete-answer model remain required.
