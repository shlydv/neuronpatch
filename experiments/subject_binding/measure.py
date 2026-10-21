"""Measure calibration on the actual merged model and retain auditable logits."""

import torch
from torch.nn import functional as F

from experiments.fine_transfer.fit import sequence


@torch.no_grad()
def calibration_logits(model, protected):
    values = []
    for ids, start, _ in protected:
        values.append(
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            .logits[0, start:]
            .float()
            .cpu()
        )
    return torch.cat(values)


def statistics(logits, reference, target_log_probabilities, lengths):
    """Recompute selection metrics from saved tensors, in float64 on CPU."""
    assert logits.ndim == 2 and logits.shape == reference.shape
    assert target_log_probabilities.ndim == 2
    assert lengths.ndim == 1 and (lengths > 0).all() and int(lengths.sum()) == len(logits)
    assert all(torch.isfinite(t).all() for t in (logits, reference, target_log_probabilities))
    kls = F.kl_div(
        logits.double().cpu().log_softmax(-1),
        reference.double().cpu().softmax(-1),
        reduction="none",
    ).sum(-1)
    return {
        "nll": float(-target_log_probabilities.min()),
        "answer_kl": kls.tolist(),
        "mean_answer_kl": sum(float(row.mean()) for row in kls.split(lengths.tolist()))
        / len(lengths),
        "max_answer_kl": float(kls.max()),
        "answer_argmax_retained": int((logits.argmax(-1) == reference.argmax(-1)).sum()),
        "calibration_prompts": len(lengths),
        "calibration_positions": len(logits),
    }


@torch.no_grad()
def measure_merged(model, tokenizer, prompts, target, protected, reference):
    probabilities = []
    for prompt in prompts:
        ids, labels = sequence(tokenizer, prompt, target, model.device)
        logits = (
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            .logits[0, -len(labels) - 1 : -1]
            .float()
        )
        probabilities.append(logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1).cpu())
    values = {
        "logits": calibration_logits(model, protected),
        "target_log_probabilities": torch.stack(probabilities),
    }
    return statistics(
        values["logits"],
        reference["logits"],
        values["target_log_probabilities"],
        reference["lengths"],
    ), values
