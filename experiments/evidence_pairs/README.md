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
model predictions. Curation selects 64 training, 32 calibration and 64 validation pairs. Every
preceding candidate has an explicit review decision; unused accepted reserves
remain excluded. Review is by the Codex assistant,
not independent human adjudication.

Every selected pair also has a separately checked question about a retained fact.
That question remains answerable under BOTH context edits. These controls expose
updates that react to incomplete-looking passages regardless of the question.
They are assistant-authored, with explicit source-unit references and canonical
answers verified to remain in both contexts.

The [complete data check](data_review_checks_v1.json) verifies **256 training
context labels**, 128 calibration labels and 256 validation labels, with 640 unique
prompt-cache IDs. Each training pair contributes two primary decisions and two
retained-fact labels. Report this full supervision budget. The largest prompt is
381 tokens; article overlap is zero. All review decisions preceded model use.
Reviewed aliases replace erroneous original gold, which is retained separately.
Surviving name mentions require explicit review of whether the asked relation survives.

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

## Frozen empirical comparison

[Protocol v1](protocol_v1.json) runs the pinned Qwen3 4B on a standard T4, capped at
7200 seconds. It first tests the calibrated base on train/calibration only. If
there are fewer than eight training missing-evidence errors or four calibrated
calibration errors, it stops before training or exposing validation predictions.

Otherwise there are 12 fits: guided rows, matched random rows and LoRA, ordinary
CE versus CE plus the pairwise term, at two shared learning rates. All adapt the
same up projections at layers 23/29/34, with exactly **184,320 FP32 trainables**:
24 rows/layer or rank-5 LoRA/layer. Two epochs use the same labels, retained-fact
CE and maximum response-token reference KL. Guided supports use training-only
pair-gradient scores and stay identical across both objectives. This is an
experimental selector, not the published FiNE selector or a NeST reproduction.

**The total CE+pair objective is not bias-invariant**: its added ranking term is,
but it still contains ordinary CE. Whether the extra term actually reduces the
refusal shortcut is an empirical question. All evaluation uses merged FP16 base
weights; row and LoRA artifacts have different factor storage dtypes.

Every method gets the same calibration-only UNKNOWN bias grid. The simple
no-training baseline gets it too, and the bias is applied only to document QA.
Calibration actually generates the forced-UNKNOWN branch and verifies the chosen
setting against direct full generation. It does not assume UNKNOWN terminates.
No validation or utility results choose rates, supports or thresholds.

[HellaSwag utility manifest](utility_manifest_v1.json) freezes 64 unrelated
questions, balanced across answer letters and source domains, one per source ID.
This generative MCQ diagnostic is not the official HellaSwag likelihood metric.
It supplies no training, replay, neuron-selection or calibration signal.

A promising development result needs at least five percentage points more correct
missing-evidence answers than the calibrated base, with at least 95% retention of
baseline-correct supported answers, retained facts and utility. Report every fit,
actual answers and evidence discrimination. A result here still needs untouched
transfer, more seeds and another model/task before any robust-library claim.

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
