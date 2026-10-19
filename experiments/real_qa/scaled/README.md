# Nested data-size benchmark preparation

Prepared from the same official SQuAD 2.0 training source and deterministic article
partitions as the 128-example pilot. No model predictions exist for the added rows.
Training sizes 128, 512 and 2048 are nested prefixes of train_expanded.json; the
original prepared/train.json matches the first 128 rows exactly. Calibration and
validation expand from 32 to 256 rows each; *_new.json excludes the previously used
32. Never describe old rows as fresh. Confirmation is unchanged and was not read
by this preparation script.

The same full-context 512-token filter, paired answerability sampling and gold
aliases apply. Raw text is CC BY-SA 4.0, not MIT; see ../DATA_LICENSE.md. The public
source may have appeared in pretraining. Hashes and counts are in manifest.json.
A data-size learning curve must compare matched model/objective settings and report
both number of examples and optimizer steps, not attribute gains from extra data
to a changed selector. This prepares a stronger benchmark; it is not a result.
