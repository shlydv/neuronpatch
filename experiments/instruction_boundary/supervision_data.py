"""Fresh document allocation with matched legitimate controls for every family."""

import json

from .authority_data import CONFIRMATION_FAMILIES
from .prepare import context_key, digest, examples


def confirmation_bank(source_rows, excluded_documents, *, count):
    if type(count) is not int or count <= 0:
        raise ValueError("Positive document count required")
    excluded = set(excluded_documents)
    groups = {}
    for row in source_rows:
        if any(
            not isinstance(row.get(k), str) or not row[k].strip()
            for k in ("context", "question", "ideal")
        ):
            raise ValueError("Nonempty source fields required")
        key = context_key(row["context"])
        if key not in excluded:
            groups.setdefault(key, []).append(row)
    keys = sorted(groups, key=lambda key: digest("supervision-confirmation-v1:" + key))[:count]
    if len(keys) != count:
        raise ValueError("Insufficient fresh source documents")
    cases, membership = [], []
    for key in keys:
        source = min(groups[key], key=lambda row: digest(json.dumps(row, sort_keys=True)))
        rows = examples(source, key, families=CONFIRMATION_FAMILIES)
        clean = rows[0]
        for family, template in CONFIRMATION_FAMILIES.items():
            rows.append(
                {
                    **clean,
                    "id": key + ":direct_full:" + family,
                    "kind": "direct_full",
                    "family": family,
                    "expected": clean["marker"],
                    "messages": [
                        clean["messages"][0].copy(),
                        {
                            "role": "user",
                            "content": clean["messages"][-1]["content"]
                            + "\n\n"
                            + template.format(marker=clean["marker"]),
                        },
                    ],
                }
            )
        cases.extend(rows)
        membership.append(
            {
                "bank": "confirmation",
                "document_id": key,
                "question_sha256": digest(source["question"]),
                "reference_sha256": digest(source["ideal"]),
            }
        )
    return {"cases": cases, "membership": membership, "excluded_documents": sorted(excluded)}
