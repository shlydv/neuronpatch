"""Guard against conflating scope leakage, answer retention and surface changes."""

from experiments.fine_transfer.scope_analysis import digest_output, scope_counts


def test_cold_digest_checks_token_identity_as_well_as_decoded_text():
    row = {"text": " answer", "tokens": [1, 2], "truncated": False}
    assert digest_output(row) == digest_output(dict(row))
    assert digest_output(row) != digest_output({**row, "tokens": [3]})
    assert digest_output(row) != digest_output({**row, "text": "different"})
    assert digest_output(row) != digest_output({**row, "truncated": True})


def test_scope_leakage_is_new_target_on_other_subject_not_any_changed_text():
    bank = {
        "new_target": "Newland",
        "probes": [
            {"source_index": i, "reference_answers": gold}
            for i, gold in enumerate(([["Oldland", "Old country"]], "Elsewhere", "Thirdland"))
        ],
    }
    before = [{"text": text} for text in ("Old country.", "Newland.", "uncertain")]
    after = [{"text": text} for text in ("Newland.", "Newland.", "Thirdland.")]
    counts, decisions = scope_counts(bank, before, after)
    assert counts["baseline_reference_prefix"] == 1
    assert counts["lost_reference_prefix"] == 1
    assert counts["retained_baseline_reference_prefix"] == 0
    assert counts["edited_reference_prefix"] == 1
    assert counts["edited_new_target_prefix"] == 2
    assert counts["new_target_leakage"] == 1
    assert counts["exact_continuation_agreement"] == 1
    assert decisions[0]["new_target_leakage"]
    assert not decisions[1]["new_target_leakage"]
