"""Hold occurrence supports fixed and preserve other subjects' answer positions."""

import math

import torch
from torch.nn import functional as F

from experiments.fine_transfer.fit import sequence
from neuronpatch.knowledge import knowledge_edit_loss
from neuronpatch.occurrence import OccurrenceColumnEdit


def prompt_ids(tokenizer, prompt, device):
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    return torch.cat([torch.tensor([[tokenizer.eos_token_id]]), ids], -1).to(device)


def answer_kl(edited, reference):
    """Mean token KL(base || edited), over the full vocabulary at each position."""
    if edited.ndim not in (1, 2) or edited.shape != reference.shape:
        raise ValueError("Matching vocabulary vectors or token matrices required")
    return (
        F.kl_div(
            F.log_softmax(edited.float(), -1),
            F.softmax(reference.detach().float(), -1),
            reduction="none",
        )
        .sum(-1)
        .mean()
    )


@torch.no_grad()
def capture_calibration(model, tokenizer, prompts, max_tokens=4):
    """Freeze base continuations as contexts; never regenerate them under a patch."""
    protected = []
    for prompt in prompts:
        ids = prompt_ids(tokenizer, prompt, model.device)
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        length = generated.shape[1] - ids.shape[1]
        assert 0 < length <= max_tokens
        context = generated[:, :-1]
        start = ids.shape[1] - 1
        logits = model(input_ids=context, attention_mask=torch.ones_like(context), use_cache=False)
        protected.append((context, start, logits.logits[0, start:].float().clone()))
    return protected


def correction(edit, examples, references, config):
    terms = []
    for (ids, labels), reference in zip(examples, references, strict=True):
        logits = edit(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
        terms.append(
            knowledge_edit_loss(
                logits.float(),
                reference,
                labels,
                edit.delta,
                alpha=config.alpha,
                beta=config.beta,
                gamma=0,
            )
        )
    nll = torch.stack([r.nll for r in terms]).max()
    prefix = torch.stack([r.kl for r in terms]).mean()
    repeat = torch.stack([r.repetition for r in terms]).mean()
    norm = edit.delta.norm() * config.gamma
    return nll + prefix + repeat + norm, {
        "nll": float(nll.detach()),
        "prefix_kl": float(prefix.detach()),
        "repetition": float(repeat.detach()),
        "norm": float(norm.detach()),
    }


@torch.no_grad()
def measure(edit, examples, references, protected, config):
    loss, values = correction(edit, examples, references, config)
    kls, per_prompt, same = [], [], []
    for ids, start, reference in protected:
        logits = edit(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[
            0, start:
        ]
        token_kl = F.kl_div(
            logits.float().log_softmax(-1), reference.softmax(-1), reduction="none"
        ).sum(-1)
        kls.extend(token_kl.tolist())
        per_prompt.append(float(token_kl.mean()))
        same.append(int((logits.argmax(-1) == reference.argmax(-1)).sum()))
    return {
        **values,
        "correction_loss": float(loss),
        "answer_kl": kls,
        "mean_answer_kl": sum(per_prompt) / len(per_prompt),
        "max_answer_kl": max(kls),
        "answer_argmax_retained": sum(same),
        "calibration_prompts": len(protected),
        "calibration_positions": len(kls),
    }


def fit_binding(model, tokenizer, prompts, target, targets, calibration, weight, config):
    """Fixed-step objective ablation; accumulate exact mean preservation gradients.

    Every condition starts from zero deltas. Unchanged weights between backwards
    make sequential calibration backwards equivalent to one summed objective
    up to floating-point rounding;
    separate graphs avoid retaining all calibration activations simultaneously.
    Calibration preserves the base distribution, not benchmark ground-truth labels.
    """
    if not calibration or len(set(calibration)) != len(calibration):
        raise ValueError("Unique nonempty calibration prompts required")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Finite nonnegative preservation weight required")
    if config.early_stop_probability is not None:
        raise ValueError("Objective comparison requires fixed steps, without early stopping")
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
    protected = capture_calibration(model, tokenizer, calibration)
    edit = OccurrenceColumnEdit(model, targets, delta_dtype=config.delta_dtype).eval()
    optimizer = torch.optim.Adam(edit.trainable_parameters(), lr=config.lr)
    history = []
    for step in range(config.max_steps):
        total, values = correction(edit, examples, references, config)
        total.backward()
        kls = []
        for ids, start, reference in protected:
            with torch.set_grad_enabled(weight > 0):
                logits = edit(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                kl = answer_kl(logits.logits[0, start:], reference)
                if weight > 0:
                    (weight * kl / len(protected)).backward()
            kls.append(float(kl.detach()))
        if edit.delta.grad is None or not torch.isfinite(edit.delta.grad).all():
            raise FloatingPointError("Invalid accumulated edit gradient")
        history.append(
            {
                "step": step + 1,
                **values,
                "mean_answer_kl": sum(kls) / len(kls),
                "max_prompt_mean_answer_kl": max(kls),
                "total": float(total.detach()) + weight * sum(kls) / len(kls),
            }
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(edit.delta).all():
            raise FloatingPointError("Nonfinite edit")
        if step == 0 or (step + 1) % 10 == 0:
            print("BIND_TRAIN", weight, step + 1, history[-1], flush=True)
    final = measure(edit, examples, references, protected, config)
    return edit, history, final
