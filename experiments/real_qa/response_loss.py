"""Avoid passage-token vocabulary logits when only response suffixes are supervised."""

import torch
import torch.nn.functional as F


def response_logits(model, inputs):
    """Return response-predicting logits and labels with identical causal alignment.

    Requires a nonempty contiguous supervised suffix in each left-padded row.
    The model must support Hugging Face's logits_to_keep argument. Every hidden
    position is still computed; only the expensive vocabulary projection is sliced.
    """
    labels = inputs["labels"]
    if labels.ndim != 2 or labels.shape != inputs["input_ids"].shape:
        raise ValueError("labels must match the two-dimensional input IDs")
    selected = labels != -100
    counts = selected.sum(dim=1)
    if not ((counts > 0) & (counts < labels.shape[1])).all():
        raise ValueError("each row needs response tokens and at least one prefix token")
    expected = torch.arange(labels.shape[1], device=labels.device)[None, :] >= (
        labels.shape[1] - counts[:, None]
    )
    if not torch.equal(selected, expected):
        raise ValueError("supervision must be a contiguous response suffix")
    width = int(counts.max().item())
    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    logits = model(**model_inputs, use_cache=False, logits_to_keep=width + 1).logits
    if logits.shape[1] != width + 1:
        raise ValueError("model did not honor logits_to_keep")
    # Include the final prefix position, exclude final EOS's unused next-token logit.
    return logits[:, :-1], labels[:, -width:]


def response_ce(logits, labels):
    """Standard token-mean causal CE, with the model's usual float32 loss precision."""
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
