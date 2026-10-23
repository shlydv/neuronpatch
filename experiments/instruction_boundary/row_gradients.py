"""Selected incoming-row gradients without retaining full projection gradients."""

import torch
from torch import nn

from neuronpatch.layers import validate_indices
from neuronpatch.losses import completion_nll


def capture_rows(model, sample, targets, *, loss_scale=128.0):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Frozen eval-mode model required")
    if not targets or sample.input_ids.shape[0] != 1 or not 0 < loss_scale < float("inf"):
        raise ValueError("Targets, single example and finite positive loss scale required")
    modules = {name: model.get_submodule(name) for name in targets}
    if any(type(m) is not nn.Linear for m in modules.values()):
        raise ValueError("Ordinary Linear targets required")
    indices = {name: validate_indices(targets[name], m.out_features) for name, m in modules.items()}
    handles, pending, gradients = [], {}, {}

    def hook(name):
        def forward(module, args, output):
            if name in pending or name in gradients:
                raise ValueError("Each projection must execute once")
            pending[name] = args[0][0].detach().float().cpu()

            def backward(derivative):
                selected = (
                    derivative[0]
                    .index_select(-1, torch.tensor(indices[name], device=derivative.device))
                    .detach()
                    .float()
                    .cpu()
                    / loss_scale
                )
                # Explicit index_select avoids advanced-index axis changes.
                if selected.shape[0] != pending[name].shape[0]:
                    raise ValueError("Token axes differ")
                value = selected.double().T @ pending.pop(name).double()
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite selected row gradients")
                gradients[name] = value

            handles.append(output.register_hook(backward))

        return forward

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        embedding = model.get_input_embeddings()(sample.input_ids).detach().requires_grad_(True)
        logits = model(
            inputs_embeds=embedding, attention_mask=sample.attention_mask, use_cache=False
        ).logits
        loss = completion_nll(
            logits, sample.input_ids, sample.completion_mask, sample.attention_mask
        )
        (loss * loss_scale).backward()
        if pending or set(gradients) != set(targets):
            raise ValueError("Missing selected-row derivatives")
        return gradients
    finally:
        for handle in handles:
            handle.remove()
