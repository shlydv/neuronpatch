"""One prediction-space objective for different trainable parameterizations."""

import math
from dataclasses import dataclass

import torch

from experiments.fine_transfer.fit import sequence
from experiments.subject_binding.fit import answer_kl, capture_calibration
from neuronpatch.knowledge import knowledge_edit_loss


@dataclass
class References:
    examples: list
    prefixes: list
    protected: list


@torch.no_grad()
def capture_references(model, tokenizer, prompts, target, calibration):
    if not prompts or not calibration or len(set(calibration)) != len(calibration):
        raise ValueError("Nonempty correction prompts and unique calibration prompts required")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Capture references on an eval-mode frozen base")
    examples = [sequence(tokenizer, p, target, model.device) for p in prompts]
    prefixes = [
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        .logits[0, : -len(labels) - 1]
        .float()
        .clone()
        for ids, labels in examples
    ]
    return References(examples, prefixes, capture_calibration(model, tokenizer, calibration))


def correction(model, references, *, alpha, beta):
    """Same target/prefix/repetition loss, without support- or factor-norm penalties.

    A zero placeholder satisfies the reference loss's occurrence argument; its
    norm coefficient is zero. LoRA factor norms and occurrence norms represent
    different quantities, so neither is used in this comparison.
    """
    terms = []
    for (ids, labels), prefix in zip(references.examples, references.prefixes, strict=True):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[
            0
        ]
        terms.append(
            knowledge_edit_loss(
                logits.float(),
                prefix,
                labels,
                logits.new_zeros((1, 1)),
                alpha=alpha,
                beta=beta,
                gamma=0,
            )
        )
    nll = torch.stack([t.nll for t in terms]).max()
    prefix = torch.stack([t.kl for t in terms]).mean()
    repetition = torch.stack([t.repetition for t in terms]).mean()
    total = nll + prefix + repetition
    return total, {
        "nll": float(nll.detach()),
        "prefix_kl": float(prefix.detach()),
        "repetition": float(repetition.detach()),
    }


def fit_shared(model, parameters, references, *, steps, lr, weight, alpha, beta, check_time):
    """Train either representation with identical accumulated prediction losses.

    FP32 trainables and fixed updates are required for every family. Sequential
    calibration backwards average prompts equally and keep one graph at a time.
    """
    parameters = list(parameters)
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Unique nonempty trainable parameters required")
    if actual != {id(p) for p in parameters} or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Supply all and only FP32 trainable parameters")
    if type(steps) is not int or steps <= 0 or not references.protected:
        raise ValueError("Positive fixed steps and preservation contexts required")
    if not all(math.isfinite(v) and v >= 0 for v in (lr, weight, alpha, beta)) or lr == 0:
        raise ValueError("Finite coefficients and positive learning rate required")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    for step in range(steps):
        check_time()
        task, values = correction(model, references, alpha=alpha, beta=beta)
        task.backward()
        kls = []
        for ids, start, reference in references.protected:
            with torch.set_grad_enabled(weight > 0):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                kl = answer_kl(logits.logits[0, start:], reference)
                if weight:
                    (weight * kl / len(references.protected)).backward()
            kls.append(float(kl.detach()))
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Missing or nonfinite trainable gradient")
        row = {
            "step": step + 1,
            **values,
            "mean_answer_kl": sum(kls) / len(kls),
            "total": float(task.detach()) + weight * sum(kls) / len(kls),
        }
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Nonfinite trained parameter")
        history.append(row)
        if step == 0 or (step + 1) % 10 == 0:
            print("SHARED_TRAIN", row, flush=True)
    return history
