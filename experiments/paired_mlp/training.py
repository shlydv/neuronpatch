"""Same complete-answer optimization and measurement for every representation."""

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
):
    """Run the fixed reference updates; optional observers must not mutate state.

    before_update sees clipped/unscaled gradients before Adam; after_update sees
    the resulting weights. Observers may measure/export but must preserve model,
    gradients, optimizer and RNG. The checkpoint study checks exact final parity.
    """
    parameters = list(parameters)
    assert parameters and {id(p) for p in parameters} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    assert all(p.dtype == torch.float32 for p in parameters)
    optimizer = torch.optim.Adam(parameters, lr=condition["lr"])
    generator = torch.Generator().manual_seed(protocol["seed"])
    orders = [
        torch.randperm(len(rows), generator=generator).tolist() for rows in (training, preserved)
    ]
    history = []
    for step in range(protocol["steps"]):
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
            logits = model(
                input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
            ).logits
            loss = completion_nll(
                logits, sample.input_ids, sample.completion_mask, sample.attention_mask
            )
            (loss * weight * protocol["loss_scale"]).backward()
            losses.append(float(loss.detach()))
            del logits, loss
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("Missing trainable gradient")
            parameter.grad.div_(protocol["loss_scale"])
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Nonfinite trainable gradient")
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
        if before_update is not None:
            before_update(step + 1, row, optimizer)
        optimizer.step()
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
