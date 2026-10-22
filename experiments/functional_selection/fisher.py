"""Frozen-reference softmax Fisher sketches for proposed neuron updates.

These measure local output-distribution curvature on supplied response contexts.
They are not gradients of KL(reference||reference), which would be zero, nor
guarantees about finite updates, new contexts or free-running generations.
"""

import math

import torch

from neuronpatch.paired import _layout


def logit_weights(logits, noise):
    """Return a detached, centered Fisher probe for equally weighted positions.

    If E[noise noise.T]=I over the flattened [positions,vocabulary] coordinates,
    E[weights weights.T] equals the block-diagonal softmax covariance divided
    by positions. Independent Rademacher noise provides this second moment.
    """
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or logits.shape[0] == 0
        or logits.shape[1] < 2
        or noise.shape != logits.shape
        or not noise.is_floating_point()
        or noise.device != logits.device
        or not torch.isfinite(logits).all()
        or not torch.isfinite(noise).all()
    ):
        raise ValueError(
            "finite floating [positions,vocabulary] logits and matching noise required"
        )
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probability = logits.detach().to(dtype).softmax(-1)
    raw = probability.sqrt() * noise.detach().to(dtype)
    centered = raw - probability * raw.sum(-1, keepdim=True)
    return centered / math.sqrt(len(logits))


def restrict_factors(captures, selections):
    """Keep candidate incoming rows/outgoing columns in their supplied order.

    Factors are [tokens,input] and [tokens,output], not dense weight gradients.
    Selection layout must already have been checked against the actual model.
    """
    rows = []
    for capture in captures:
        result = {}
        for selection in selections:
            indices = torch.tensor(selection.indices, dtype=torch.long)
            for name in selection.incoming:
                inputs, derivative = capture[name]
                result[name] = (inputs, derivative.index_select(1, indices))
            inputs, derivative = capture[selection.outgoing]
            result[selection.outgoing] = (inputs.index_select(1, indices), derivative)
        rows.append(result)
    return rows


def capture_fisher_factors(model, example, selections, *, probes, seed, loss_scale=128.0):
    """One forward and several local-noise VJPs; return compact CPU factors.

    The original base is frozen and in eval mode. Each projection must execute
    once over one complete sequence. All token positions contribute indirectly
    through the graph; only response prediction positions seed the VJP.
    Prompt/padding logits receive no direct probe. No parameter .grad or caller
    RNG is changed, and hooks are removed before returning or raising.
    """
    if (
        model.training
        or any(p.requires_grad for p in model.parameters())
        or not torch.is_grad_enabled()
        or type(probes) is not int
        or probes < 1
        or type(seed) is not int
        or seed < 0
        or isinstance(loss_scale, bool)
        or not math.isfinite(loss_scale)
        or loss_scale <= 0
    ):
        raise ValueError("eval-mode frozen base,positive probe count/scale and local seed required")
    ids, attention, response = (
        example.input_ids,
        example.attention_mask,
        example.completion_mask,
    )
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or ids.shape[1] < 2
        or ids.dtype != torch.long
        or any(
            m.shape != ids.shape or m.dtype != torch.bool or m.device != ids.device
            for m in (attention, response)
        )
        or response[:, 0].any()
        or (response & ~attention).any()
        or (response[:, 1:] & ~attention[:, :-1]).any()
        or not response.any()
    ):
        raise ValueError("one attended sequence with explicitly aligned response targets required")
    checked = _layout(model, tuple(selections))
    if len({module.weight.device for module, _, _ in checked.values()}) != 1:
        raise ValueError("single-device projection layout required")
    activations, outputs, handles = {}, {}, []

    def capture(name, axis, indices):
        def hook(module, args, output):
            if name in outputs or args[0].shape[:2] != ids.shape:
                raise ValueError("each projection must execute once over the complete sequence")
            inputs = args[0][0].detach()
            if axis == "input":
                inputs = inputs.index_select(1, indices)
            activations[name] = inputs.float().cpu()
            outputs[name] = output

        return hook

    try:
        for name, (module, axis, indices) in checked.items():
            index_tensor = torch.tensor(indices, dtype=torch.long, device=module.weight.device)
            handles.append(module.register_forward_hook(capture(name, axis, index_tensor)))
        embedding = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        logits = model(inputs_embeds=embedding, attention_mask=attention, use_cache=False).logits
        selected = logits[0, :-1][response[0, 1:]]
        if set(outputs) != set(checked):
            raise ValueError("a projection was not executed")
        generator = torch.Generator(device=selected.device).manual_seed(seed)
        result = []
        for probe in range(probes):
            noise = torch.empty_like(selected, dtype=torch.float32)
            noise.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
            weights = logit_weights(selected, noise)
            objective = (selected.to(weights.dtype) * weights).sum()
            derivatives = torch.autograd.grad(
                objective * loss_scale,
                tuple(outputs[name] for name in checked),
                retain_graph=probe < probes - 1,
            )
            factors = {}
            for (name, (_, axis, indices)), derivative in zip(
                checked.items(), derivatives, strict=True
            ):
                values = derivative[0].detach().float() / loss_scale
                if axis == "output":
                    index_tensor = torch.tensor(indices, dtype=torch.long, device=values.device)
                    values = values.index_select(1, index_tensor)
                values = values.cpu()
                if not torch.isfinite(values).all() or not torch.isfinite(activations[name]).all():
                    raise FloatingPointError("nonfinite projection factors")
                factors[name] = (activations[name], values)
            result.append(factors)
        return result
    finally:
        for handle in handles:
            handle.remove()
