"""Per-token output-contribution localization from FiNE's knowledge editor.

Scores each FFN feature by activation times its outgoing vector's alignment with
an unembedding vector. This selects supports; it does not train an editor or
establish preservation. The reference editor scores its original predicted token
at each teacher-forced target position, not necessarily the desired new token.
"""

from collections.abc import Sequence

import torch


@torch.no_grad()
def contribution_indices(
    activations: torch.Tensor,
    down_weights: Sequence[torch.Tensor],
    unembedding: torch.Tensor,
    predicted_tokens: torch.Tensor,
    k: int,
    *,
    exclude_last_layers: int = 3,
    score_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return [positions, k, 2] CPU indices, with each pair (layer, feature).

    activations is [positions, layers, FFN width]; each down weight is PyTorch's
    [hidden, FFN width]; unembedding is [vocabulary, hidden]. Scores use the
    unembedding device. FP16 matches the upstream FiNE scoring precision. A
    caller may explicitly request FP32/FP64; that can change rankings.

    Excluded layers retain their original layer numbering. Repeated selections
    across positions are retained: deduplication can change the reference
    editor's parameterization and regularization. Ties follow torch.topk and
    may differ across devices/runtime versions. All source tensors are detached.
    """
    if activations.ndim != 3 or not activations.is_floating_point():
        raise ValueError("activations must be floating [positions, layers, FFN width]")
    positions, layers, width = activations.shape
    if min(positions, layers, width) <= 0:
        raise ValueError("activation dimensions must be nonempty")
    if unembedding.ndim != 2 or not unembedding.is_floating_point():
        raise ValueError("unembedding must be floating [vocabulary, hidden]")
    vocabulary, hidden = unembedding.shape
    if min(vocabulary, hidden) <= 0:
        raise ValueError("unembedding dimensions must be nonempty")
    if len(down_weights) != layers or any(
        weight.shape != (hidden, width) or not weight.is_floating_point() for weight in down_weights
    ):
        raise ValueError("one floating [hidden, FFN width] down weight is required per layer")
    if type(exclude_last_layers) is not int or not 0 <= exclude_last_layers < layers:
        raise ValueError("exclude_last_layers must leave at least one eligible layer")
    eligible = layers - exclude_last_layers
    if type(k) is not int or not 1 <= k <= eligible * width:
        raise ValueError("k must fit the eligible neuron count")
    if predicted_tokens.shape != (positions,) or predicted_tokens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("predicted_tokens must be an integer vector, one per position")
    if predicted_tokens.device.type == "meta":
        raise ValueError("all tensors must be materialized")
    tokens = predicted_tokens.detach().cpu().tolist()
    if min(tokens) < 0 or max(tokens) >= vocabulary:
        raise ValueError("predicted token outside vocabulary")
    if score_dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("score_dtype must be float16, float32 or float64")
    device = unembedding.device
    if (
        device.type == "meta"
        or activations.device.type == "meta"
        or any(weight.device.type == "meta" for weight in down_weights)
    ):
        raise ValueError("all tensors must be materialized")
    result = []
    for position, token in enumerate(tokens):
        direction = unembedding[token].to(device=device, dtype=score_dtype)
        scores = torch.empty((eligible, width), device=device, dtype=score_dtype)
        for layer in range(eligible):
            activation = activations[position, layer].to(device=device, dtype=score_dtype)
            outgoing = down_weights[layer].to(device=device, dtype=score_dtype)
            scores[layer] = activation * (outgoing.T @ direction)
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite contribution scores; check inputs and scoring precision")
        flat = torch.topk(scores.flatten(), k).indices
        result.append(torch.stack((flat // width, flat % width), dim=-1).cpu())
    return torch.stack(result)
