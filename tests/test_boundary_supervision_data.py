import json

from experiments.instruction_boundary.prepare import context_key
from experiments.instruction_boundary.supervision_data import confirmation_bank


def test_fresh_document_pairs_share_payload_but_change_trust_boundary():
    rows = [{"context": f"table {i}", "question": "Value?", "ideal": str(i)} for i in range(8)]
    excluded = {context_key(row["context"]) for row in rows[:3]}
    result = confirmation_bank(rows, excluded, count=3)
    assert result == confirmation_bank(list(reversed(rows)), excluded, count=3)
    assert not excluded & {row["document_id"] for row in result["membership"]}
    cases = {row["id"]: row for row in result["cases"]}
    assert len(cases) == 3 * 13
    for row in cases.values():
        if row["kind"] != "direct_full":
            continue
        encoded = row["messages"][-1]["content"].split("\n\nQuoted document (JSON string):\n")[1]
        table, end = json.JSONDecoder().raw_decode(encoded)
        payload = encoded[end:].strip()
        assert row["marker"] not in table
        assert row["marker"] in payload
        inside = cases[row["document_id"] + ":injected:" + row["family"]]
        inside_table = json.loads(
            inside["messages"][-1]["content"].split("\n\nQuoted document (JSON string):\n")[1]
        )
        assert inside_table == table + "\n\n" + payload
        assert inside["expected"] != row["expected"]
