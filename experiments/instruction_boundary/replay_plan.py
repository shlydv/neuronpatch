"""Validate comparator selection across one or more immutable parent runs."""


def parents(spec):
    if "reused_parent" in spec and "reused_parents" in spec:
        raise ValueError("Use one replay configuration form")
    result = spec.get("reused_parents", [spec["reused_parent"]] if "reused_parent" in spec else [])
    labels = [label for parent in result for label in choices(parent)]
    if len(labels) != len(set(labels)):
        raise ValueError("Comparator labels must be unique across parents")
    fingerprints = {parent["base_fingerprint"] for parent in result}
    if len(fingerprints) > 1:
        raise ValueError("All comparator parents must share the base model")
    return result


def choices(parent):
    selected = parent.get("replay_choices", parent["choices"])
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected) <= set(parent["choices"])
    ):
        raise ValueError("Replay choices must be a nonempty unique subset of frozen choices")
    return selected
