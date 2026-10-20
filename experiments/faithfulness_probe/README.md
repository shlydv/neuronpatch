# External development probe: conditional use of supplied evidence

The [previous controlled abstention test](../evidence_pairs/colab_v1/REPORT.md)
stopped before adaptation because the unchanged 4B model was already near ceiling.
This probe establishes baseline headroom on published FaithEval examples before
investing in another training recipe. It makes no method-improvement claim.

## Frozen design

[Protocol](protocol_v1.json), [exact source revisions/IDs](data_manifest_v1.json).
Pinned Qwen3-4B-Instruct-2507, FP16, greedy generation, 300 queries, at most
2048 prompt-plus-output tokens and 1800 seconds of model execution. No weights
change. No examples are truncated to fit.

All 1,000 FaithEval counterfactual questions and choices match original ARC
records exactly. The probe selects 48 questions, four per distinct combination
of original-science answer and fabricated-context answer. Both answer-letter
distributions are balanced. For each question it evaluates:

- Ordinary instruction to answer from the passage.
- Explicit instruction to report the passage's claim even if scientifically false.
- Instruction to answer using established science despite the untrusted passage.
- Original science question without the passage.

Twenty-four cases, selected before predictions, also rotate the answer choices
under both context-reporting instructions. The gold letters rotate with the
answer texts. This checks whether a purported failure or fix depends on a letter.
It does not alter the question or supplied scientific claims.

Another 30 unanswerable and 30 inconsistent examples cover all ten source subsets,
three per subset. Both use the same prompt, which asks for UNKNOWN if evidence is
missing and CONFLICT if relevant evidence conflicts. Task names, answers and
author justifications are never supplied to the model.

The initial 1536-token preparation cap admitted only two inconsistent SearchQA
examples. It was raised to 2048 using input-length coverage, before any frozen
dataset or model predictions. Actual maximum input length is 1998 tokens. This
length-limited, balanced subset is a development diagnostic, **not an official
full-benchmark FaithEval score**.

## Interpretation before adaptation

Use complete generated answers for primary results. Do not replace them with
probabilities renormalized over the four choice letters. Strict parsing is
reported alongside an assistant review of semantic failures, expanded answers,+ambiguous questions and source-label problems. Broad substring rules such as
counting every response containing “not” as abstention are prohibited.

Before adapting the reporting behavior, require at least eight of the 48 cases
to fail semantically despite a correct closed-book science answer, with at least
four still failing under explicit reporting instructions. Alternatively require
at least six reviewed semantic errors among 30 on another task. These are
feasibility criteria, not proof of method superiority. Unexpected findings are
reported as exploratory.

An edit that improves document reporting but destroys science answers in the
other modes is insufficient. First establish the behavior that needs changing
and a no-training prompt control. Then use separate training material and fresh
confirmation cases; never fine-tune on these benchmark test inputs. Unselected
FaithEval IDs remain unused for later evaluation. This development slice becomes
exposed after the run.

## Sources and reconstruction

Public author datasets:
[FaithEval](https://github.com/SalesforceAIResearch/FaithEval),
[counterfactual](https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0),
[unanswerable](https://huggingface.co/datasets/Salesforce/FaithEval-unanswerable-v1.0),
[inconsistent](https://huggingface.co/datasets/Salesforce/FaithEval-inconsistent-v1.0),
[original ARC](https://huggingface.co/datasets/allenai/ai2_arc).
The ARC card lists CC BY-SA 4.0. The inspected FaithEval cards describe a research
release without a standalone dataset-license field; the repository's Apache-2.0
code license is not asserted to relicense its source datasets. Raw passages stay
in ignored local artifacts; public manifests contain revision pins, hashes and IDs.

Download the pinned source parquet files and cards in the manifest, preserving
the `<dataset-name>/<file>` layout under `--source`, and store their records as
`download_manifest.json`. With the pinned tokenizer cached:

```sh
python -m experiments.faithfulness_probe.prepare --source SOURCE --output DATA
python -m experiments.faithfulness_probe.build --data DATA --output BUNDLE
```

No historical private directories are required. The generated Colab cell bundles
exact source/data bytes, verifies a SHA256 before execution, streams logs and
archives outputs even on failure. Colab remains the authorized fallback for the
cancelled Kaggle queue; execution here must not be labeled a Kaggle result.
