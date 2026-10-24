"""Matched discovery prompts: same payload, different authority boundary.

Experimental data construction, not a trained selector or a security guarantee.
Both members retain the full table and question. Only one moves the redirect
instruction outside the quoted JSON document, where the user may request it.
Use training documents only; no reference answers or generated responses enter.
"""

import json

from .prepare import FAMILIES, SYSTEM


def matched_authority_pairs(training_cases):
    """Build two labeled prompts per clean training case, in source order.

    label1: ignore the instruction inside untrusted data and answer the question.
    label0: follow the user's instruction outside the data and return its marker.
    Labels describe intended authority, not the base model's observed behavior.
    The caller must enforce train/calibration/confirmation membership boundaries.
    """
    clean = [row for row in training_cases if row["kind"] == "clean"]
    if not clean or len({r["document_id"] for r in clean}) != len(clean):
        raise ValueError("One nonempty clean case per unique training document required")
    result = []
    separator = "\n\nQuoted document (JSON string):\n"
    for row in clean:
        messages = row["messages"]
        if (
            len(messages) != 2
            or messages[0] != {"role": "system", "content": SYSTEM}
            or messages[1]["role"] != "user"
            or row["family"] != "none"
        ):
            raise ValueError("Expected frozen clean boundary prompt")
        prefix, boundary, encoded = messages[1]["content"].partition(separator)
        if not boundary or not prefix.startswith(
            "Answer this question using the quoted document: "
        ):
            raise ValueError("Expected question and JSON document boundary")
        document = json.loads(encoded)
        if not isinstance(document, str) or not document or row["marker"] in document:
            raise ValueError("Nonempty string document without reserved marker required")
        payload = FAMILIES["redirect"].format(marker=row["marker"])
        for label, authority in ((1, "untrusted"), (0, "user")):
            content = (
                prefix + separator + json.dumps(document + "\n\n" + payload, ensure_ascii=False)
                if label
                else prefix
                + separator
                + json.dumps(document, ensure_ascii=False)
                + "\n\n"
                + payload
            )
            result.append(
                {
                    "id": row["document_id"] + ":authority_pair:" + authority,
                    "document_id": row["document_id"],
                    "source_case_id": row["id"],
                    "label": label,
                    "authority": authority,
                    "messages": [messages[0].copy(), {"role": "user", "content": content}],
                }
            )
    return result
