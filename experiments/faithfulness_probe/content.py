"""Conservative post-run option-content extraction, separate from frozen scoring.

Only a bare choice, an exact option text, or a matching letter+exact option text
is resolved. Extra explanations, corrections and letter/content disagreements
require explicit review. The method never consults which option is gold.
"""

import re


def normalized(text):
    return " ".join(text.strip().strip("\"'").rstrip(".!?").casefold().split())


def exact_option(text, option_texts):
    """Return an unambiguous A-D choice based on exact supplied option content."""
    labels = list("ABCD")
    if len(option_texts) != 4:
        raise ValueError("Four ordered options required")
    text = text.strip()
    bare = re.fullmatch(r"\(?([A-D])\)?[.!?)]?", text)
    if bare:
        return bare.group(1)
    prefix = re.match(r"^\(?([A-D])\)?[.)]\s+(.+)$", text, re.S)
    body = prefix.group(2) if prefix else text
    matches = [
        labels[i] for i, value in enumerate(option_texts) if normalized(body) == normalized(value)
    ]
    if len(matches) != 1 or (prefix and prefix.group(1) != matches[0]):
        return None
    return matches[0]
