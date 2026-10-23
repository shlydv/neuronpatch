"""Same complete-answer optimization and measurement for every representation."""

import math

import torch

from neuronpatch.losses import completion_nll


def fit(
    model,
    parameters,
    training,
    preserved,
    protocol,
    condition,
    check_time,
    *,
    before_update=None,
    after_update=None,
    auxiliary_loss=None,
    after_backward=None,
):
    """Run the fixed reference updates; optional observers must not mutate state.

    before_update sees clipped/unscaled gradients before Adam; after_update sees
    the resulting weights. Observers may measure/export but must preserve model,
    gradients, optimizer and RNG. The checkpoint study checks exact final parity.
    An optional auxiliary_loss(index,sample) supplies a scalar objective for the
    auxiliary bank only. It must not step parameters or mutate model state.
    The default canonical-response path and all optimizer operations are unchanged.
    after_backward(step,bank) may observe cumulative scaled gradients after each
    bank's backward pass; it must not mutate parameters, gradients or model state.
    With minimum_loss_scale, GradScaler backs off on nonfinite gradients and
    retries the same examples before any Adam update. No scale growth occurs.
    Retries require deterministic eval-mode computation without callbacks.
    """
    parameters = list(parameters)
    assert parameters and {id(p) for p in parameters} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    assert all(p.dtype == torch.float32 for p in parameters)
    optimizer = torch.optim.Adam(parameters, lr=condition["lr"])
    minimum_scale = protocol.get("minimum_loss_scale")
    scaler = None
    if minimum_scale is not None:
        if (
            not math.isfinite(minimum_scale)
            or minimum_scale <= 0
            or minimum_scale > protocol["loss_scale"]
        ):
            raise ValueError("Invalid minimum loss scale")
        if model.training or any(
            x is not None for x in (before_update, after_update, auxiliary_loss, after_backward)
        ):
            raise ValueError(
                "Scale retries require eval-mode training without callback side effects"
            )
        scaler = torch.amp.GradScaler(
            parameters[0].device.type,
            init_scale=protocol["loss_scale"],
            growth_interval=2**30,
            backoff_factor=0.5,
        )
    generator = torch.Generator().manual_seed(protocol["seed"])
    orders = [
        torch.randperm(len(rows), generator=generator).tolist() for rows in (training, preserved)
    ]
    history = []
    for step in range(protocol["steps"]):
        check_time()
        retries = []
        while True:
            check_time()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            indices = []
            for bank, rows, weight in (
                (0, training, 1.0),
                (1, preserved, protocol["preservation_weight"]),
            ):
                index = orders[bank][step % len(rows)]
                indices.append(index)
                sample = rows[index]
                if bank == 1 and auxiliary_loss is not None:
                    loss = auxiliary_loss(index, sample)
                else:
                    logits = model(
                        input_ids=sample.input_ids,
                        attention_mask=sample.attention_mask,
                        use_cache=False,
                    ).logits
                    loss = completion_nll(
                        logits, sample.input_ids, sample.completion_mask, sample.attention_mask
                    )
                    del logits
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite forward loss at step {step + 1}, bank {bank}, example {index}"
                    )
                weighted = loss * weight
                (
                    scaler.scale(weighted)
                    if scaler is not None
                    else weighted * protocol["loss_scale"]
                ).backward()
                if after_backward is not None:
                    after_backward(step + 1, bank)
                losses.append(float(loss.detach()))
                del loss, weighted
            if any(p.grad is None for p in parameters):
                raise RuntimeError("Missing trainable gradient")
            if scaler is None:
                for parameter in parameters:
                    parameter.grad.div_(protocol["loss_scale"])
            else:
                scaler.unscale_(optimizer)
            if all(torch.isfinite(p.grad).all() for p in parameters):
                break
            scale = protocol["loss_scale"] if scaler is None else scaler.get_scale()
            if scaler is None or scale / 2 < minimum_scale:
                raise FloatingPointError(
                    f"Nonfinite trainable gradient at step {step + 1}, "
                    f"examples {indices}, losses {losses}, loss_scale {scale}"
                )
            retry = {
                "step": step + 1,
                "loss_scale": scale,
                "example_indices": indices,
                "losses": losses,
            }
            print("GRADIENT_RETRY", condition["id"], retry, flush=True)
            retries.append(retry)
            # GradScaler skips Adam on nonfinite gradients, then reduces scale.
            # Recompute the same examples; do not advance the data or step count.
            scaler.step(optimizer)
            scaler.update()
        norm = torch.nn.utils.clip_grad_norm_(parameters, protocol["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite global gradient norm")
        row = {
            "step": step + 1,
            "task_nll": losses[0],
            "preserve_nll": losses[1],
            "gradient_norm": float(norm),
            "example_indices": indices,
        }
        if scaler is not None:
            row["loss_scale"] = scaler.get_scale()
            row["gradient_retries"] = retries
        if before_update is not None:
            before_update(step + 1, row, optimizer)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Nonfinite trainable value")
        history.append(row)
        if step == 0 or (step + 1) % 10 == 0:
            print("TRAIN", condition["id"], row, flush=True)
        if after_update is not None:
            after_update(step + 1, row, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return history


@torch.no_grad()
def teacher(model, examples):
    values, labels, lengths = [], [], []
    for sample in examples:
        output = model(
            input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
        ).logits
        mask = sample.completion_mask[0, 1:]
        values.append(output[0, :-1][mask].detach().cpu())
        labels.append(sample.input_ids[0, 1:][mask].cpu())
        lengths.append(int(mask.sum()))
    return {
        "logits": torch.cat(values),
        "labels": torch.cat(labels),
        "lengths": torch.tensor(lengths),
    }


def teacher_metrics(tensors, task_count, preserved_count):
    logits = tensors["logits"].double()
    labels = tensors["labels"]
    nll = logits.logsumexp(-1) - logits.gather(1, labels[:, None])[:, 0]
    assert torch.isfinite(nll).all()
    chunks = nll.split(tensors["lengths"].tolist())
    means = [float(c.mean()) for c in chunks]
    offset = task_count + preserved_count
    return {
        "task_mean_nll": sum(means[:task_count]) / task_count,
        "task_worst_token_nll": max(float(c.max()) for c in chunks[:task_count]),
        "preserve_mean_nll": sum(means[task_count:offset]) / preserved_count,
        "target_val_mean_nll": sum(means[offset:]) / len(means[offset:]),
    }
