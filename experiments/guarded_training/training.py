"""Full-response fitting with measured, base-anchored preservation acceptance.

Support selection and adapter representation are supplied by the caller. The
same loop can train paired-neuron patches or official LoRA. Held-out evaluation
stays outside this loop, and no pretrained advantage is claimed.
"""

import math
import time

import torch

from neuronpatch.losses import completion_nll

from .step import Budget, Measurement, backtracked_step


def reference_kl(current, reference):
    """Forward-KL loss with exactly zero autograd gradient at the reference.

    The log-sum-exp Bregman form avoids separately rounded softmax and
    log-softmax paths creating a spurious initial preservation gradient. Actual
    acceptance still uses an independent FP64 probability-distribution KL.
    """
    reference = reference.detach().to(device=current.device, dtype=current.dtype)
    normalizer = reference.logsumexp(-1)
    probability = (reference - normalizer[:, None]).exp()
    return (
        current.logsumexp(-1) - normalizer - (probability * (current - reference)).sum(-1)
    ).mean()


def response_logits(model, sample):
    logits = model(
        input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
    ).logits
    return logits[0, :-1][sample.completion_mask[0, 1:]]


@torch.no_grad()
def references(model, preserved):
    if model.training or not preserved:
        raise ValueError("eval-mode model and preservation responses required")
    return [response_logits(model, sample).detach().float().cpu() for sample in preserved]


@torch.no_grad()
def measure(model, training, preserved, base):
    """Measure full training NLL and exact KL(base||current) on supplied contexts."""
    losses, worst_nll, divergences = [], [], []
    for sample in training:
        logits = response_logits(model, sample).detach().double().cpu()
        labels = sample.input_ids[0, 1:][sample.completion_mask[0, 1:]].cpu()
        nll = logits.logsumexp(-1) - logits.gather(1, labels[:, None])[:, 0]
        losses.append(float(nll.mean()))
        worst_nll.append(float(nll.max()))
    for sample, old in zip(preserved, base, strict=True):
        current = response_logits(model, sample).detach().double().cpu()
        if old.shape != current.shape:
            raise ValueError("preservation reference/token layout changed")
        logp, logq = old.double().log_softmax(-1), current.log_softmax(-1)
        kl = (logp.exp() * (logp - logq)).sum(-1)
        if not torch.isfinite(kl).all() or float(kl.min()) < -1e-10:
            raise FloatingPointError("invalid measured KL")
        divergences.append(kl.clamp_min(0))
    return (
        Measurement(
            sum(losses) / len(losses),
            sum(float(k.mean()) for k in divergences) / len(divergences),
            max(float(k.max()) for k in divergences),
        ),
        max(worst_nll),
    )


def fit_guarded(
    model,
    parameters,
    training,
    preserved,
    base,
    *,
    learning_rate,
    budget: Budget,
    steps=60,
    preservation_weight=1.0,
    loss_scale=128.0,
    gradient_clip=1.0,
    confidence=0.9,
    scales=(1.0, 0.5, 0.25, 0.125),
    check_time=lambda: None,
):
    """Full-batch task CE + anchored KL; actual measurements decide acceptance.

    The confidence stop concerns every supervised correction token, including
    EOS. It does not inspect held-out completions. Budget limits apply to the
    absolute drift from the frozen base, not merely the previous iterate. No
    accepted update is a valid outcome, rather than permission to exceed limits.
    """
    parameters = list(parameters)
    if (
        model.training
        or not training
        or not preserved
        or len(base) != len(preserved)
        or type(steps) is not int
        or steps < 1
        or not 0 < confidence < 1
        or any(not math.isfinite(v) or v <= 0 for v in (learning_rate, loss_scale, gradient_clip))
        or not math.isfinite(preservation_weight)
        or preservation_weight < 0
    ):
        raise ValueError(
            "eval-mode model, responses, references and positive fitting settings required"
        )
    if {id(p) for p in parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("optimizer must contain exactly the trainable adapter parameters")
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("nonempty FP32 adapter parameters required")
    started = time.monotonic()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    before, worst = measure(model, training, preserved, base)
    initial = before
    history = []
    threshold = -math.log(confidence)
    reason = "training_confidence" if worst <= threshold else "maximum_steps"
    for step in range(1, steps + 1):
        if worst <= threshold:
            break
        check_time()
        optimizer.zero_grad(set_to_none=True)
        for sample in training:
            check_time()
            logits = model(
                input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
            ).logits
            loss = completion_nll(
                logits, sample.input_ids, sample.completion_mask, sample.attention_mask
            )
            (loss * loss_scale / len(training)).backward()
        if preservation_weight:
            for sample, old in zip(preserved, base, strict=True):
                check_time()
                current = response_logits(model, sample).float()
                kl = reference_kl(current, old)
                (kl * preservation_weight * loss_scale / len(preserved)).backward()
        for p in parameters:
            if p.grad is None:
                raise RuntimeError("disconnected adapter parameter")
            p.grad.div_(loss_scale)
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite adapter gradient")
        last_worst = worst

        def measurement():
            nonlocal last_worst
            check_time()
            result, last_worst = measure(model, training, preserved, base)
            return result

        result = backtracked_step(optimizer, measurement, before, budget, scales=scales)
        before = result["measurement"]
        history.append(
            {
                "step": step,
                "accepted": result["accepted"],
                "attempts": result["attempts"],
                "gradient_norm": float(norm),
                "worst_training_token_nll": last_worst if result["accepted"] else worst,
            }
        )
        if not result["accepted"]:
            reason = "no_accepted_update"
            break
        worst = last_worst
        print("GUARDED_STEP", step, history[-1], flush=True)
        if worst <= threshold:
            reason = "training_confidence"
    optimizer.zero_grad(set_to_none=True)
    return {
        "initial": initial.__dict__,
        "final": before.__dict__,
        "history": history,
        "accepted_updates": sum(r["accepted"] for r in history),
        "stop_reason": reason,
        "worst_training_token_nll": worst,
        "elapsed_seconds": time.monotonic() - started,
    }
