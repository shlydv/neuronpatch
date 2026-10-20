"""Development-only multi-prompt control with fixed occurrence supports.

All selectors use the identical objective, training prompts, optimizer and budget.
Only the separately labeled source-style single-prompt reference uses less data.
"""

import math

import torch

from neuronpatch.knowledge import knowledge_edit_loss
from neuronpatch.occurrence import OccurrenceColumnEdit


def sequence(tokenizer, prompt, target, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    labels = tokenizer.encode(" " + target, add_special_tokens=False, return_tensors="pt")
    ids = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), prompt_ids, labels], -1).to(device)
    return ids, labels[0].to(device)


def fit(model, tokenizer, prompts, target, targets, config):
    """Worst prompt/token NLL, mean prefix KL/repetition, one global delta norm."""
    examples = [sequence(tokenizer, p, target, model.device) for p in prompts]
    references = []
    with torch.no_grad():
        for ids, labels in examples:
            references.append(
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                .logits[0, : -len(labels) - 1]
                .float()
                .clone()
            )
    edit = OccurrenceColumnEdit(model, targets, delta_dtype=config.delta_dtype).eval()
    optimizer = torch.optim.Adam(edit.trainable_parameters(), lr=config.lr)
    history, updates = [], 0
    for step in range(config.max_steps):
        terms = []
        for (ids, labels), reference in zip(examples, references, strict=True):
            logits = (
                edit(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                .logits[0]
                .float()
            )
            terms.append(
                knowledge_edit_loss(
                    logits,
                    reference,
                    labels,
                    edit.delta,
                    alpha=config.alpha,
                    beta=config.beta,
                    gamma=0,
                )
            )
        nll = torch.stack([term.nll for term in terms]).max()
        kl = torch.stack([term.kl for term in terms]).mean()
        repetition = torch.stack([term.repetition for term in terms]).mean()
        norm = torch.norm(edit.delta, p=2) * config.gamma
        total = nll + norm + kl + repetition
        history.append(
            {
                "step": step + 1,
                "total": float(total.detach()),
                "nll": float(nll.detach()),
                "per_prompt_nll": [float(term.nll.detach()) for term in terms],
                "kl": float(kl.detach()),
                "repetition": float(repetition.detach()),
                "norm": float(norm.detach()),
            }
        )
        if step == 0 or (step + 1) % 10 == 0:
            print("TRAIN", step + 1, "max_nll", float(nll.detach()), flush=True)
        if config.early_stop_probability is not None and nll < -math.log(
            config.early_stop_probability
        ):
            break
        total.backward()
        if not torch.isfinite(edit.delta.grad).all():
            raise FloatingPointError("Nonfinite update gradient")
        optimizer.step()
        optimizer.zero_grad()
        updates += 1
        if not torch.isfinite(edit.delta).all():
            raise FloatingPointError("Nonfinite update")
    return edit, history, updates
