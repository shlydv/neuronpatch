# Experimental evidence-conditioned adaptation

This follows the [dose diagnostic](../protected_rows/dose_v1/REPORT.md): our
learned updates increased refusals without improving measured evidence
discrimination. A constant UNKNOWN logit bias captured part of the benefit.
The next experiment must distinguish conditional improvement from this shortcut.
Nothing here is a validated recipe or a new ranking algorithm.

## Objective implemented

For the same question, a supported context has answer A and a missing context
should produce UNKNOWN. At the first generated position define

\[
s^\pm=z^\pm_{\text{UNKNOWN}}-z^\pm_{A_1},\qquad d=s^- - s^+.
\]

Here A₁ is the same canonical answer token in both contexts. The implementation
in [objective.py](objective.py) uses ordinary pairwise logistic ranking plus a
penalty for movement of each pair's center:

\[
L=\mathbb E\left[\operatorname{softplus}\left((\gamma-d)/T\right)\right]
 +\lambda\mathbb E\left[(\overline s-\overline s_{\rm base})^2\right].
\]

Adding any vocabulary bias shared within a pair adds b_UNKNOWN−b_A₁ to both
scores. It cancels from d exactly in real arithmetic, even when the bias depends
on the question. Starting at the reference, it cannot lower the total loss:
ranking is unchanged and centering adds a nonnegative penalty. Tests check this,
the gradient directions, numerical derivatives and FP16-to-FP32 arithmetic.

This is a limited claim. The objective does not guarantee a correct full answer,
improved UNKNOWN-versus-all ranking, transfer or preservation. A bias can remove
center drift after conditional training. A context-dependent shortcut may still
win, and first-token answer proxies may miss alternate valid answer openings.
Full generation and simpler calibrated controls remain necessary.

## Data construction and review

Both variants remove one unit from an unused SQuAD training paragraph. Supported
removes an unrelated unit of similar length; missing removes the answer-bearing
unit. The question stays identical. Candidate prompt lengths differ by at most
eight tokens. Train/calibration/validation article sets are disjoint; paragraph
contexts used in earlier experiments are excluded. Topics may have historical
development exposure, so this is not fresh confirmation.

The controlled development subset uses concrete who/when/where facts. It is an
instrument for testing conditional learning, not representative broad QA or a
smaller definition of the library goal. [FaithEval](https://github.com/SalesforceAIResearch/FaithEval)
and official SQuAD development remain unused for later transfer evaluation.

**Lexical absence is insufficient.** The [review](review_v1.json) rejects surviving
aliases, valid inferences, ambiguous dates, false premises and malformed source
questions. Canonical forms/aliases are corrected from the source before any
model predictions. Current training curation selects 64 pairs from 100 reviewed
candidates; one further accepted case is reserve. Review is by the Codex assistant,
not independent human adjudication.

Every selected pair also has a separately checked question about a retained fact.
That question remains answerable under BOTH context edits. These controls expose
updates that react to incomplete-looking passages regardless of the question.
They are assistant-authored, with explicit source-unit references and canonical
answers verified to remain in both contexts.

The [training check](training_review_checks.json) verifies 64 articles, 128
distinct questions and **256 context labels**: 128 primary and 128 retained-fact
labels. All 256 prompt-cache IDs differ, and the largest prompt is 344 tokens.
Report this full supervision budget; calling it just 64 examples would hide data.
Calibration and validation reviews are not complete. No model has run on this data.

## Reproduce preparation

Use the pinned official SQuAD2 training source (SHA in prepare.py) and the
published reconstruction specification. This needs no historical local
directories or tokenizer. Finalization additionally needs the locally cached,
pinned Qwen3 tokenizer. Raw passages stay under ignored artifacts/.

~~~sh
python -m experiments.evidence_pairs.reconstruct \
  --source /path/to/train-v2.0.json \
  --spec experiments/evidence_pairs/candidate_spec_v1.json \
  --output artifacts/evidence_pairs/candidates3

# Only after all source-only reviews are explicitly signed off:
python -m experiments.evidence_pairs.finalize \
  --source /path/to/train-v2.0.json \
  --candidates artifacts/evidence_pairs/candidates3 \
  --review experiments/evidence_pairs/review_v1.json \
  --output artifacts/evidence_pairs/frozen1
~~~

The reconstructed files must match every original candidate hash exactly.
The original sampling script prepare.py additionally depends on the historical
files recorded in the manifest. Use it to create a new pool, not to reconstruct
version 1 from a different local history.
Finalization checks candidate/source hashes, explicit decisions for every case
before the cutoff, both context edits, retained control evidence, aliases, unique
cache IDs, token budgets and article separation. It refuses partial curation
before creating model inputs. Passing data checks is not evidence of model
quality. The next model run needs a frozen protocol and a baseline with headroom.

## Next empirical comparison

Finish 32 calibration and 64 validation pairs with retained-fact controls. Freeze
the data, inspect the baseline and establish whether clear errors leave headroom.
Compare calibrated no-training bias, ordinary supervised adaptation and the new
objective with equal data and preservation signals. Keep guided, matched-random
and suitable low-rank parameterizations separate from the objective comparison.
Do not credit localization for a benefit caused by more labels, different target
layers, parameter counts or a different objective.

Report raw and calibrated decisions, correction/preservation curves, actual
answers, retained-fact controls, unrelated utility and full cost. A promising
development result still needs untouched transfer data, more seeds and another
model/task. The current protected-row and mathematical results do not meet that
requirement.

## Relevant established work

Pairwise logistic ranking is established; see
[From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf).
Our use of it is a task-specific hypothesis, not algorithmic novelty.
[Two Axes of LLM Abstention](https://arxiv.org/html/2607.08456v1) distinguishes
answerability and answer correctness and exposes surface confounds; its results
are not a reproduction of our document-grounding task.
[Context-DPO](https://github.com/byronbbl/Context-DPO) already trains
context-faithful preferences. Its implementation has not been ported or evaluated
here; it is relevant prior work for a later trained baseline.
