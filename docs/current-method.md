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

The [objective audit](../experiments/paired_endpoint/objective_diagnosis_v1.json)
finds only5/6 preservation-training references initially matched in each case.
The sixth also imposes a supervised correction or wording constraint. In the
high-rate outgoing-only condition,17 of22 scored test losses become an exact
answer found in these auxiliary training examples, versus zero that become the
main edited target. This is consistent with broader training-answer spillover;
it is not causal attribution to a particular example. It also shows why counting
only the injected target string would miss much of the observed damage.

The [evaluation-contract audit](../experiments/paired_endpoint/evaluation_audit_v1.json)
also flags equivalent Unicode formulas, valid full extensions to fragment prompts,
and single-country labels for ambiguous association questions. For example,
[FIFA names three host countries](https://inside.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/media-releases/canada-mexico-and-usa-selected-as-hosts-of-the-2026-fifa-world-cuptm),
while our legacy validation row accepts only Mexico. These are reasons to fix
the next calibration contract, not to rewrite the frozen scores or dismiss
unambiguous interference. Future trials should distinguish factual content
accuracy from the requested response form.

The [completed eight-endpoint replay](../experiments/paired_endpoint/complete_v1/REPORT.md)
resolves the deployment uncertainty: all248 live/merged answer pairs are exactly
identical. The high-rate failure and selected low-rate interference already
exist in the saved live patches. The exact final sampled correction prompt also
degrades between its logged pre-update state and the saved endpoint. Checkpoint
traces now isolate that late failure to an optimizer update: grad-enabled and
no-grad logits agree at the measured before/after states, while the update
changes a correct word continuation into premature EOS. Merging is not the
source of the observed answer failures.

The [completed seven-fit checkpoint study](../experiments/paired_checkpoints/complete_v1/REPORT.md)
selects guided checkpoints at12 updates for both facts, versus30/24 for random
and LoRA. All three early methods transfer8/8 answers. However, the guided
checkpoint retains both the Lakers target leakage and oxygen-symbol error;
early random and early LoRA have no scored preservation losses. Guided total
adaptation prefix is144.89seconds across two cases, versus154.28 for random and
37.84 for LoRA. Sparse attachment and export each take about30seconds per case
and contain strict full-base integrity checks. Earlier stopping is useful
control but does not rescue this recipe.

## What the papers suggest, and what we actually adopted

- **FiNE:** its inspected [knowledge-editor implementation](https://github.com/opanhw/fine-unified-editing-code/blob/a2489cbac0dfe5a956385734065cfa041384e7d1/knowledge-editing/easyeditor/models/fine/fine_main.py)
  combines a worst-target-token loss, prefix-distribution KL, update-norm and
  repetition penalties with a probability stopping threshold. The pinned
  [GPT-J configuration](https://github.com/opanhw/fine-unified-editing-code/blob/a2489cbac0dfe5a956385734065cfa041384e7d1/knowledge-editing/hparams/FINE/gptj-6b.yaml)
  sets50 maximum epochs and early-stop probability0.9. It selects five
  occurrences per target token before flattening, rather than173 indices spread
  over every layer. Our initial comparison used fixed50 updates, mean full-response
  CE and preservation examples; it did not reproduce this objective or budget.
  The completed follow-up tests complete-response stopping under the initial
  objective; it does not reproduce FiNE's other loss terms. Stopping and norm
  control are existing design choices, not new inventions.
- **NeST:** probe-based localization and clustered incoming-neuron updates
  motivate controlling which activation conditions change and how much capacity
  is shared. Its official archive is now retrieved and inspected; the
  [source fidelity audit](../experiments/nest_source/README.md) checks the actual
  shared-row operation against ours on CPU. Full discovery and safety training
  remain unreproduced. The released trainer's default differs from the paper's
  epoch count, and padding, threshold and judge fixtures expose distinctions that
  a comparison must record. Coupled rows/columns alone do not reproduce NeST.
- **SafeNexus:** exclusion of normal-prompt candidates motivates explicitly
  asking which proposed updates overlap preserved behavior. Activation exclusion
  is not itself a guarantee of functional preservation. Its implementation
  remains unverified; do not present our gradient score as its selector. Its
  [Appendix3 utility argument](https://arxiv.org/html/2607.28969v2) assumes low
  utility-gradient sensitivity in the selected subspace. That assumption needs
  measurement in our setting; neuron sparsity alone does not establish it.
- **SpaRTA and LoRA:** random sparse adaptation and low-rank adaptation are
  strong controls, not straw baselines. Our whole-neuron random control is not
  SpaRTA's random scalar support. Discovery and tuning costs must count against
  a guided method, even if its optimizer takes fewer steps.

See [research provenance](research.md) for source and reproduction limits. The
bounded original FiNE comparison did reproduce a selected-versus-random gap on
original answer prefixes; it did not establish robust complete-answer editing.

## Next candidate and decision boundary

Do not advance the current gradient-magnitude selector unchanged into fresh
confirmation. It learns the training answers quickly, but its smaller earlier
patch still changes unrelated facts. The next selection hypothesis must measure
correction gain together with functional sensitivity on representative
preservation inputs, rather than treating correction-gradient concentration as
evidence of safe update directions. Enforce the same measured preservation and
stopping policy for guided supports, random supports and LoRA. Retain a no-edit
outcome when no candidate satisfies it. Multiple parameter budgets remain
necessary so an easy transfer ceiling does not hide an efficiency difference.

The [functional-direction candidate](../experiments/functional_selection/README.md)
now estimates conditional Fisher disturbance for explicit, unit-normalized
fresh-Adam neuron proposals. It compares gradient magnitude, proposed gain,
gain/risk and random supports. Its first bounded calibration uses sixteen
restored one-step edits before a longer optimizer fit: the local proxy must
predict finite changes and improve useful learning per unit held-out drift.
CPU covariance/dense-gradient oracles and FP32/FP16 tiny-Qwen execution pass;
there is no new pretrained outcome yet. This stage does not establish a robust
recipe, and it does not replace the required trained LoRA comparison.
Its [frozen Kaggle calibration](../experiments/functional_selection/launch_v1.json)
is accepted as version1 and running, with byte-identical accepted source. No
other experiment is submitted alongside it.

The [no-forward context audit](../experiments/functional_selection/context_audit_v1.json)
also bounds interpretation of that calibration. Equally averaging its six
canonical anchors,75.56% and80.00% of prediction contexts match the respective
greedy base paths. Correct shorter aliases account for some differences;
context divergence is not itself a factual error. Canonical-context drift
remains a valid conditional measurement, but it does not cover all ordinary
generated-response contexts. Future preservation comparisons must specify this
choice and retain direct complete-answer evaluation. The pending experiment's
contexts, interventions and decision criteria remain frozen.

The [saved-history diagnosis](../experiments/paired_endpoint/history_diagnosis_v1.json)
suggests that guided low-rate fits cross a rough sampled-loss threshold before
random/LoRA fits. Those historical losses were observed at different pre-update
states. The new post-update study confirms faster fitting under a simultaneous
confidence gate, but not cheaper total adaptation than LoRA or removal of the
two genuine preservation errors. Optimizer-step counts alone would have given a
misleading positive conclusion.

This is a testable synthesis of localization, limited capacity, stopping and
functional protection. Its ingredients already have prior art. Our proposed
contribution must earn its value through a reproducible advantage in correction,
preservation, data or total cost. A true advantage still needs fresh cases,
another task family and model-size checks before becoming the library default.

Both the endpoint replay and frozen seven-fit checkpoint study are complete.
The latter independently verifies434 decoded answers,107 teacher records, all
seven original final controls and the recorded final Adam update. All14
selected/final grad-enabled-versus-no-grad comparisons have identical logits at
446 response positions. Its high-rate instability is distinct from the
low-rate semantic interference, which is already present at12 updates. Accepted
source and launch provenance remain in the [launch record](../experiments/paired_checkpoints/launch_v1.json).

A [retrospective audit of all24 saved weight updates](../experiments/paired_mlp/complete_v1/UPDATE_AUDIT.md)
also rules out update size alone as a sufficient explanation for the observed
interference. In case1 at the low rate, the guided norm is2.138, below clean
random seed11's4.360 and LoRA's2.269, yet the guided patch causes the genuine
Lakers and oxygen errors. Norm control may still help within a trajectory; a
global small-norm rule cannot replace functional preservation measurements.

The [saved discovery-score audit](../experiments/paired_mlp/complete_v1/DISCOVERY_AUDIT.md)
finds that the existing layer quotas retain95.52–97.17% of the same-budget
global optimum for our own correction-gradient score. Guided supports contain
about49% of the squared correction gradient, versus0.016–0.036% for the three
recorded random controls. This is not a behavioral or Adam-step advantage. It
weakens poor allocation as an explanation for wasting this particular score and
keeps the focus on whether correction sensitivity translates into useful cost
and preservation behavior.
The large [method ledger](method-design.md) preserves historical decisions; this
document is the current decision, not another claim of completed research.
