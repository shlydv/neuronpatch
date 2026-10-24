"""Matched trusted/untrusted preference pairs without answer-reference labels.

The same chosen targets also define the ordinary SFT control. This constructor
is not SecAlign's dataset recipe: it adds explicit reversed trusted preferences
so globally suppressing the marker cannot satisfy both sides. It uses cached
clean-model answers and inherits their factual mistakes. Callers must supply a
training-only membership and freeze templates separately from held-out tests.
"""

import hashlib
import json
from string import Formatter


def build_pairs(clean_cases, teachers, templates, *, allowed_documents):
    """Return two preference rows per document/template and clean preservation.

    Only canonical clean JSON-quotation prompts are accepted. The payload's text
    and response alternatives are identical on both sides; its trust location
    changes, and chosen/rejected swap. No confirmation labels are consumed.
    """
    allowed = set(allowed_documents)
    if not clean_cases or not allowed or not templates:
        raise ValueError("Nonempty clean cases, allowed membership and templates required")
    if any(
        not isinstance(name, str)
        or not name
        or ":" in name
        or not isinstance(template, str)
        or not template.strip()
        or [field for _, field, _, _ in Formatter().parse(template) if field is not None]
        != ["marker"]
        for name, template in templates.items()
    ):
        raise ValueError("Named templates must contain exactly one marker placeholder")
    if len({r["id"] for r in clean_cases}) != len(clean_cases):
        raise ValueError("Unique clean case IDs required")
    outputs = {r["id"]: r for r in teachers}
    if len(outputs) != len(teachers) or set(outputs) != {r["id"] for r in clean_cases}:
        raise ValueError("Exactly one teacher per clean case required")
    docs = [r["document_id"] for r in clean_cases]
    if len(set(docs)) != len(docs) or set(docs) != allowed:
        raise ValueError("Exactly one clean case per allowed document required")
    pairs, clean, excluded = [], [], []
    for case in clean_cases:
        if case["kind"] != "clean" or case["family"] != "none":
            raise ValueError("Only clean source cases are allowed")
        messages = case["messages"]
        if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
            raise ValueError("Canonical system/user prompt required")
        prefix, separator, encoded = messages[-1]["content"].partition(
            "\n\nQuoted document (JSON string):\n"
        )
        if not separator or not prefix.startswith(
            "Answer this question using the quoted document: "
        ):
            raise ValueError("Canonical clean quotation required")
        document = json.loads(encoded)
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Nonempty quoted source required")
        teacher = outputs[case["id"]]
        if type(teacher["ended"]) is not bool or not isinstance(teacher["text"], str):
            raise ValueError("Explicit teacher text and termination required")
        if not teacher["ended"] or not teacher["text"].strip():
            excluded.append(case["document_id"])
            continue
        answer, marker = teacher["text"], case["marker"]
        if (
            not isinstance(marker, str)
            or not marker.strip()
            or marker in answer
            or marker in messages[-1]["content"]
        ):
            raise ValueError("Marker must be distinct from source and teacher response")
        clean.append(
            {
                "id": case["id"],
                "document_id": case["document_id"],
                "messages": [dict(m) for m in messages],
                "target": answer,
            }
        )
        for name, template in templates.items():
            payload = template.format(marker=marker)
            for trusted in (False, True):
                quoted = document if trusted else document + "\n\n" + payload
                user = prefix + separator + json.dumps(quoted, ensure_ascii=False)
                if trusted:
                    user += "\n\n" + payload
                pairs.append(
                    {
                        "id": case["document_id"]
                        + ":"
                        + ("trusted" if trusted else "untrusted")
                        + ":"
                        + name,
                        "document_id": case["document_id"],
                        "template": name,
                        "trusted": trusted,
                        "messages": [dict(messages[0]), {"role": "user", "content": user}],
                        "chosen": marker if trusted else answer,
                        "rejected": answer if trusted else marker,
                        "teacher_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    }
                )
    return {"pairs": pairs, "clean": clean, "excluded_documents": sorted(excluded)}
