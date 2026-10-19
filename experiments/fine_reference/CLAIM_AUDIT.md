# What the FiNE comparison should establish

Audited 2026-10-19. These are published figures, not our results.

[Supplement, Table XIII](https://opanhw.github.io/fine-unified-editing/static/pdfs/supplementFile.pdf)
reports the following GPT-J comparison (percent):

| Metric | Selected | Random (without localization) |
|---|---:|---:|
| Edit success | 99.8 | 85.6 |
| Subject aliasing | 90.6 | 60.8 |
| Logical generalization | 17.5 | 16.0 |
| Reasoning | 37.4 | 27.9 |
| Relation-specific locality | 84.2 | 85.0 |
| Forgetfulness locality | 54.2 | 63.8 |

Selection improves correction and several transfer metrics; it does not dominate
preservation. This ablation is not itself a training-data-size curve. Do not
translate it into a universal fewer-examples guarantee. Sixteen cases can check
execution and explore the effect, but cannot reproduce full-benchmark numbers.

Supplement SHA256:
`e10f936ab21d113ed09488790cd444bc302e386aad25d34ebae0becf972c3bbd`.

## Source integration issue

At the pinned commit, `knowledge-editing/run.py` defines `--random` but never
reads `args.random` afterward. The actual `fine_main.py` branches on
`hparams.random`. Running the provided driver twice with and without that CLI
flag would therefore not establish a selected/random contrast. Our wrapper
assigns `hp.random` explicitly before calling the unchanged editor, and records
the condition in each output. This source issue does not prove which commands
the authors used for the table.

## Prepared subset coverage

The frozen first 16 cases have 130 relation-specific locality probes, 7
forgetfulness probes, 61 reasoning probes, 94 alias probes, and 15 logical
probes. All nested first answers are nonempty. These unequal counts make one
pooled all-probe score misleading. Report categories separately and preserve
per-case results; specify weighting and missing-category treatment before
comparison to the full paper. In particular, seven forgetfulness probes are weak
evidence for that dimension.

## Consequence for our library

A candidate selector must earn its cost through correction, transfer, or a
better preservation trade-off at matched budgets. It need not win every metric.
Keep distinct: ability to execute an edit, reproduction of published metrics,
quality of generated answers, and useful library-level improvement. Neither
CPU checks nor favorable selected examples establish the latter three.
