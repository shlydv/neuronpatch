"""Guard against conflating scope leakage, answer retention and surface changes."""

import json
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("corruption", ["subject", "duplicate_fit", "cold", "prefilled_answer"])
def test_scope_integrity_rejects_balanced_but_unpaired_or_incomplete_outputs(tmp_path, corruption):
    """A correct row count must not conceal a changed subject, patch or control."""
    from experiments.fine_transfer.verify_scope import check_results

    here = Path(__file__).parents[1] / "experiments/fine_transfer"

    def read(name):
        return json.loads((here / name).read_text())

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    protocol, banks, cold, primed = [
        read(name)
        for name in (
            "scope_execution_v1.json",
            "scope_inputs_v1.json",
            "scope_cold_v1.json",
            "scope_primed_v1.json",
        )
    ]
    outputs = [{"text": "placeholder", "tokens": [1], "truncated": False} for _ in range(8)]
    completion = {
        "status": "complete",
        "new_fits": 0,
        "cold_checks_before_new_probes": True,
        "weights_restored": True,
        "final_fingerprint": protocol["base_fingerprint"],
        "conditions": [],
    }
    checks = []
    for case in cold:
        i = case["case"]
        write(f"case-{i}-base.json", {"bank": banks[i], "outputs": outputs})
        for row in case["conditions"]:
            name = row["condition"]
            completion["conditions"].append({"case": i, "condition": name})
            checks.append({"case": i, "condition": name, "checked": row["expected"]})
            write(
                f"case-{i}-{name}.json",
                {"case": i, "condition": name, "bank": banks[i], "after": outputs},
            )
        for name in ("base", "consensus"):
            write(
                f"case-{i}-primed-{name}.json",
                {
                    "probes": [p for p in primed if p["case"] == i],
                    "tails": outputs[:3],
                    "answer_provided_in_input": True,
                },
            )
    write("completion.json", completion)
    write("cold-checks.json", checks)
    write(
        "identity.json",
        {"protocol": protocol, "fingerprint": protocol["base_fingerprint"], "torch": "2.4.1"},
    )
    assert check_results(tmp_path, protocol, banks, cold, primed)["output_sequences_checked"] == 108
    if corruption == "duplicate_fit":
        completion["conditions"][-1] = completion["conditions"][0]
        write("completion.json", completion)
    elif corruption == "cold":
        # Same count, but one replay came from the wrong saved patch.
        replacement = json.loads((tmp_path / "cold-checks.json").read_text())
        replacement[0]["checked"] = replacement[1]["checked"]
        write("cold-checks.json", replacement)
    else:
        name = "case-0-fine-original.json" if corruption == "subject" else "case-0-primed-base.json"
        replacement = json.loads((tmp_path / name).read_text())
        if corruption == "subject":
            replacement["bank"]["probes"][0]["subject"] = "A different person"
        else:
            replacement["answer_provided_in_input"] = False
        write(name, replacement)
    with pytest.raises(AssertionError):
        check_results(tmp_path, protocol, banks, cold, primed)
