"""Exact probability of a finite set of complete autoregressive responses.

This is ordinary marginal likelihood, not a semantic judge. Callers supply the
accepted token paths and logits for every needed prediction prefix. EOS makes
the complete paths disjoint; aliases are summed rather than averaged. Missing
prefixes are errors, never guessed probabilities. No model is invoked here.
"""

from collections.abc import Iterable, Mapping

import torch

TokenPath = tuple[int, ...]


def _paths(responses: Iterable[TokenPath], eos_token_id: int) -> tuple[TokenPath, ...]:
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("one nonnegative integer EOS token required")
    paths = tuple(dict.fromkeys(tuple(p) for p in responses))
    if not paths or any(
        len(p) < 2
        or any(type(t) is not int or t < 0 for t in p)
        or p[-1] != eos_token_id
        or eos_token_id in p[:-1]
        for p in paths
    ):
        raise ValueError("nonempty content followed by exactly one final EOS required")
    return paths


def covered_paths(
    contexts: Mapping[TokenPath, torch.Tensor],
    responses: Iterable[TokenPath],
    *,
    eos_token_id: int,
) -> tuple[tuple[TokenPath, ...], tuple[TokenPath, ...]]:
    """Partition paths for an explicitly partial, lower-bound diagnostic.

    This function alone does not compute a probability. Training callers should
    use log_response_mass on their full accepted set, which rejects missing data.
    """
    present, missing = [], []
    for path in _paths(responses, eos_token_id):
        target = present if all(path[:i] in contexts for i in range(len(path))) else missing
        target.append(path)
    return tuple(present), tuple(missing)


def log_response_mass(
    contexts: Mapping[TokenPath, torch.Tensor],
    responses: Iterable[TokenPath],
    *,
    eos_token_id: int,
) -> torch.Tensor:
    """Return log P(response in the supplied finite set), with autograd.

    Each context maps a *response* prefix (empty at the first token) to a
    one-dimensional next-token logit vector for the same conditioning prompt.
    All vectors must share vocabulary, device and floating dtype. Duplicated
    response paths count once. No length normalization, greedy decoding,
    reference-token masking or alias-frequency weighting is applied.
    """
    paths = _paths(responses, eos_token_id)
    probabilities = {}
    signature = None
    for path in paths:
        for i, token in enumerate(path):
            prefix = path[:i]
            if prefix not in contexts:
                raise ValueError(f"missing prediction prefix: {prefix}")
            logits = contexts[prefix]
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 1
                or not logits.numel()
                or not logits.is_floating_point()
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("finite floating one-dimensional logit vectors required")
            current = (logits.numel(), logits.dtype, logits.device)
            if signature is not None and current != signature:
                raise ValueError("prediction contexts must share vocabulary, dtype and device")
            signature = current
            if token >= logits.numel():
                raise ValueError("response token outside vocabulary")
            if prefix not in probabilities:
                values = (
                    logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
                )
                probabilities[prefix] = values.log_softmax(-1)
    return torch.logsumexp(
        torch.stack(
            [
                torch.stack([probabilities[p[:i]][t] for i, t in enumerate(p)]).sum()
                for p in paths
            ]
        ),
        dim=0,
    )
