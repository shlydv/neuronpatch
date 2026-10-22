# Current method decision — 22 October 2026

We can train small patches that change complete answers. We have not shown
that selecting neurons by our gradient score produces a better adaptation than
matched random neurons or LoRA. The useful research question is selective,
efficient adaptation with measured preservation, rather than making an already
learned target answer even more confident.

## Evidence that changes the next action

The [completed Qwen3-4B comparison](../experiments/paired_mlp/complete_v1/REPORT.md)
matches approximately1.33million trainable scalars and reports all rates and
three random seeds. Every selected method gives8/8 complete transfer answers
across two exposed facts. The gradient-paired condition leaks its injected
country into an unrelated Lakers answer. One random condition has no measured
test preservation loss; selected LoRA's one exact-score loss is a valid formal
UK name. These observations provide no selection advantage.

Near-zero preservation-training CE is not a safety measurement on other inputs.
The damaging paired case has lower preservation CE than the clean random case.
The scope-validation bank also starts with only1/6 accepted answers, so its
"lose no correct answers" rule offers weak coverage. Future protection needs
unambiguous, representative questions and review of genuine factual errors
separately from wording and output-form changes.

One endpoint uncertainty remains: high-rate paired case0 is broken after
merging, but its final live predictions were not saved. The
[frozen eight-endpoint replay](../experiments/paired_endpoint/README.md) tests
both states using exactly the saved deltas and existing questions, with zero
new fits. Do not infer an optimizer or merging cause before it completes.

## What the papers suggest, and what we actually adopted

- **FiNE:** its inspected [knowledge-editor implementation](https://github.com/opanhw/fine-unified-editing-code/blob/a2489cbac0dfe5a956385734065cfa041384e7d1/knowledge-editing/easyeditor/models/fine/fine_main.py)
  combines a worst-target-token loss, prefix-distribution KL, update-norm and
  repetition penalties with a probability stopping threshold. The pinned
  [GPT-J configuration](https://github.com/opanhw/fine-unified-editing-code/blob/a2489cbac0dfe5a956385734065cfa041384e7d1/knowledge-editing/hparams/FINE/gptj-6b.yaml)
  sets50 maximum epochs and early-stop probability0.9. It selects five
  occurrences per target token before flattening, rather than173 indices spread
  over every layer. Our new comparison used fixed50 updates, mean full-response
  CE and preservation examples; it did not reproduce this objective or budget.
  Stopping once the correction is adequate and controlling update size are
  existing design choices worth testing, not new inventions.
- **NeST:** probe-based localization and clustered incoming-neuron updates
  motivate controlling which activation conditions change and how much capacity
  is shared. This remains paper-derived guidance; its implementation has not
  been verified here. Coupled rows/columns alone do not reproduce its clustering.
- **SafeNexus:** exclusion of normal-prompt candidates motivates explicitly
  asking which proposed updates overlap preserved behavior. Activation exclusion
  is not itself a guarantee of functional preservation. Its implementation
  remains unverified; do not present our gradient score as its selector.
- **SpaRTA and LoRA:** random sparse adaptation and low-rank adaptation are
  strong controls, not straw baselines. Our whole-neuron random control is not
  SpaRTA's random scalar support. Discovery and tuning costs must count against
  a guided method, even if its optimizer takes fewer steps.

See [research provenance](research.md) for source and reproduction limits. The
bounded original FiNE comparison did reproduce a selected-versus-random gap on
original answer prefixes; it did not establish robust complete-answer editing.

## Next candidate and decision boundary

After resolving endpoint fidelity, test the smallest update that reaches a
complete-answer target, with functional preservation checked on representative
calibration examples. Compare the same stopping/protection procedure for guided
supports, random supports and LoRA. Investigate multiple parameter budgets so
a transfer ceiling does not hide a possible efficiency difference. Retain a
no-edit outcome when no candidate meets the measured preservation constraints.

The [saved-history diagnosis](../experiments/paired_endpoint/history_diagnosis_v1.json)
suggests that guided low-rate fits cross a rough sampled-loss threshold before
random/LoRA fits. Those losses were observed at different pre-update states;
they do not prove simultaneous convergence, cheaper total compute or that early
stopping prevents interference. A new experiment must measure the actual
endpoint, discovery cost, calibration cost and deployment behavior.

This is a testable synthesis of localization, limited capacity, stopping and
functional protection. Its ingredients already have prior art. Our proposed
contribution must earn its value through a reproducible advantage in correction,
preservation, data or total cost. A true advantage still needs fresh cases,
another task family and model-size checks before becoming the library default.

No new training comparison is launched while the endpoint replay is pending.
The large [method ledger](method-design.md) preserves historical decisions; this
document is the current decision, not another claim of completed research.
