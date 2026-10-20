"""Mean-gradient and coherence controls for a future selector comparison.

This does not change the frozen pilot or its frozen follow-up. All statistics
reuse the same reverse-mode factors; no additional model passes are required.
"""

import torch

from .gradient import capture_gradient_factors


def gradient_statistics(factors):
    """Compute actual column-gradient statistics without dense weight gradients.

    For prompt i, G_i[:, j] = B_i.T @ A_i[:, j]. The average norm control ranks
    ||mean_i G_i[:, j]||. The worst-gain candidate ranks the minimum first-order
    reduction over prompts along that normalized mean. Coherence reports
    ||mean G|| / mean ||G|| and does not establish semantic specificity.

    Prompt weighting is uniform, including different sequence lengths. Factors
    must already correspond to the same scalar-loss definition and scale.
    """
    if not factors:
        raise ValueError("At least one prompt required")
    activations, derivatives = [], []
    for inputs, outputs in factors:
        if (
            inputs.ndim != 2
            or outputs.ndim != 2
            or inputs.shape[0] != outputs.shape[0]
            or min(*inputs.shape, *outputs.shape) <= 0
        ):
            raise ValueError("Matching nonempty token matrices required")
        activations.append(inputs.detach().double().cpu())
        derivatives.append(outputs.detach().double().cpu())
    if len({a.shape[1] for a in activations}) != 1 or len({b.shape[1] for b in derivatives}) != 1:
        raise ValueError("Widths must agree across prompts")
    if not all(torch.isfinite(x).all() for x in activations + derivatives):
        raise ValueError("Finite factors required")
    a, b = torch.cat(activations), torch.cat(derivatives)
    projected = (b @ b.T) @ a / len(factors)
    inner_products, prompt_norms = [], []
    start = 0
    for inputs, outputs in zip(activations, derivatives, strict=True):
        end = start + len(inputs)
        inner_products.append((inputs * projected[start:end]).sum(0))
        prompt_norms.append((inputs * ((outputs @ outputs.T) @ inputs)).sum(0).clamp_min(0).sqrt())
        start = end
    inner_products = torch.stack(inner_products)
    norms = torch.stack(prompt_norms)
    mean_norm = inner_products.mean(0).clamp_min(0).sqrt()
    mean_prompt_norm = norms.mean(0)
    worst_gain = torch.zeros_like(mean_norm)
    nonzero = mean_norm > 0
    worst_gain[nonzero] = inner_products[:, nonzero].min(0).values / mean_norm[nonzero]
    coherence = torch.zeros_like(mean_norm)
    nonzero_prompts = mean_prompt_norm > 0
    coherence[nonzero_prompts] = mean_norm[nonzero_prompts] / mean_prompt_norm[nonzero_prompts]
    return {
        "mean_gradient_norm": mean_norm,
        "worst_prompt_gain": worst_gain,
        "prompt_gradient_norms": norms,
        "coherence": coherence,
    }


def discover_controls(model, examples, names, *, neurons_per_token, loss_scale=128):
    """Select single/mean/worst supports from one shared set of backward passes.

    examples contains (input_ids [1, sequence], target_tokens [target_length])
    with the same target tokens appended to each prompt. Names are ordered linear
    projections. Repeated columns across target positions remain occurrences,
    matching the pilot's parameter accounting. No fitting or generation occurs.
    """
    if not examples or not names or len(set(names)) != len(names):
        raise ValueError("Nonempty examples and unique projection names required")
    widths = [model.get_submodule(name).in_features for name in names]
    if type(neurons_per_token) is not int or not 0 < neurons_per_token <= sum(widths):
        raise ValueError("Neuron budget outside available projection columns")
    labels = examples[0][1]
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Nonempty one-dimensional target tokens required")
    for ids, target in examples:
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or ids.shape[1] <= len(labels)
            or not torch.equal(target, labels)
            or not torch.equal(ids[0, -len(labels) :], labels)
        ):
            raise ValueError("Every example must append the same target tokens to a prompt")
    names_by_column = [
        (name, feature) for name, width in zip(names, widths) for feature in range(width)
    ]
    supports = {name: [] for name in ("single_gradient", "mean_gradient", "worst_gradient")}
    records = []
    for position in range(len(labels)):
        captured, losses = [], []
        for ids, _ in examples:
            factors, loss = capture_gradient_factors(
                model,
                ids,
                ids.shape[1] - len(labels) - 1 + position,
                int(labels[position]),
                names,
                loss_scale=loss_scale,
            )
            captured.append(factors)
            losses.append(loss)
        statistics = [gradient_statistics([row[name] for row in captured]) for name in names]
        flat = {
            name: torch.cat([r[name] for r in statistics])
            for name in ("mean_gradient_norm", "worst_prompt_gain", "coherence")
        }
        flat["single_norm"] = torch.cat([r["prompt_gradient_norms"][0] for r in statistics])
        for selector, score in (
            ("single_gradient", "single_norm"),
            ("mean_gradient", "mean_gradient_norm"),
            ("worst_gradient", "worst_prompt_gain"),
        ):
            indices = torch.topk(flat[score], neurons_per_token).indices.tolist()
            targets = [names_by_column[i] for i in indices]
            supports[selector].extend(targets)
            records.append(
                {
                    "target_position": position,
                    "selector": selector,
                    "targets": targets,
                    "prompt_losses": losses,
                    "selected_statistics": {
                        name: values[indices].tolist() for name, values in flat.items()
                    },
                }
            )
    return {
        "supports": supports,
        "records": records,
        "backward_passes": len(examples) * len(labels),
        "new_fits": 0,
    }
