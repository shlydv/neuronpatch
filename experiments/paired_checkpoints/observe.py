"""Non-mutating measurements of post-update model states and the final Adam step."""

import math
import time

import torch
from safetensors.torch import save_file

from experiments.paired_endpoint.run import comparison
from experiments.paired_mlp.run import write
from experiments.paired_mlp.training import teacher


def token_losses(tensors):
    logits = tensors["logits"].double()
    return logits.logsumexp(-1) - logits.gather(1, tensors["labels"][:, None])[:, 0]


def mode_comparison(with_grad, no_grad):
    result = comparison(with_grad, no_grad)
    result["mean_kl_grad_to_no_grad"] = result.pop("mean_kl_live_to_merged")
    result["mean_kl_no_grad_to_grad"] = result.pop("mean_kl_merged_to_live")
    return result


def calibration(tensors, task_count, probability):
    losses = token_losses(tensors)
    assert torch.isfinite(losses).all()
    chunks = losses.split(tensors["lengths"].tolist())
    assert 0 < task_count < len(chunks)
    task_max = max(float(p.max()) for p in chunks[:task_count])
    preserved_max = max(float(p.max()) for p in chunks[task_count:])
    bound = -math.log(probability)
    return {
        "task_worst_token_nll": task_max,
        "preserve_worst_token_nll": preserved_max,
        "task_mean_nll": sum(float(p.mean()) for p in chunks[:task_count]) / task_count,
        "preserve_mean_nll": sum(float(p.mean()) for p in chunks[task_count:])
        / (len(chunks) - task_count),
        "adequate": task_max <= bound and preserved_max <= bound,
    }


def grad_teacher(model, samples):
    """Use the training autograd path, then detach without accumulating gradients."""
    logits, labels, lengths = [], [], []
    with torch.enable_grad():
        for sample in samples:
            output = model(
                input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
            ).logits
            mask = sample.completion_mask[0, 1:]
            logits.append(output[0, :-1][mask].detach().cpu())
            labels.append(sample.input_ids[0, 1:][mask].cpu())
            lengths.append(int(mask.sum()))
            del output
    return {
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "lengths": torch.tensor(lengths),
    }


def _sync(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Observer:
    def __init__(
        self,
        model,
        parameters,
        task,
        preserved,
        protocol,
        prefix,
        output,
        save_patch,
        check_time,
        *,
        diagnostic=False,
    ):
        self.model = model
        self.parameters = list(parameters)
        names = {id(p): n for n, p in model.named_parameters()}
        self.parameter_names = [names[id(p)] for p in self.parameters]
        self.task = task
        self.measured = [*task, *preserved]
        self.protocol = protocol
        self.prefix = prefix
        self.output = output
        self.save_patch = save_patch
        self.check_time = check_time
        self.diagnostic = diagnostic
        self.snapshots = []
        self.selected_step = None
        self.training_seconds = 0.0
        self.monitoring_seconds = 0.0
        self.serialization_seconds = 0.0
        self.diagnostic_seconds = 0.0
        self.last_exit = time.monotonic()
        self.pre_diagnostic_seconds = 0.0
        self.last_step_state = None
        self.last_step = None

    def before_update(self, step, row, optimizer):
        self.pre_diagnostic_seconds = 0.0
        if not self.diagnostic or step != self.protocol["steps"]:
            return
        _sync(self.model)
        started = time.monotonic()
        index = row["example_indices"][0]
        sample = self.task[index]
        before_grad = grad_teacher(self.model, [sample])
        before_no_grad = teacher(self.model, [sample])
        for state, tensor in (("grad", before_grad), ("no-grad", before_no_grad)):
            save_file(tensor, self.output / f"{self.prefix}-before-final-{state}.safetensors")
        state = {}
        for i, p in enumerate(self.parameters):
            state[f"parameter-before/{i}"] = p.detach().cpu().clone()
            state[f"clipped-joint-gradient/{i}"] = p.grad.detach().cpu().clone()
            saved = optimizer.state[p]
            state[f"adam-first-moment/{i}"] = (
                saved.get("exp_avg", torch.zeros_like(p)).cpu().clone()
            )
            state[f"adam-second-moment/{i}"] = (
                saved.get("exp_avg_sq", torch.zeros_like(p)).cpu().clone()
            )
        # Recompute just the selected correction gradient without touching .grad.
        from neuronpatch.losses import completion_nll

        with torch.enable_grad():
            logits = self.model(
                input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
            ).logits
            loss = completion_nll(
                logits, sample.input_ids, sample.completion_mask, sample.attention_mask
            )
            gradients = torch.autograd.grad(loss * self.protocol["loss_scale"], self.parameters)
        for i, (p, g) in enumerate(zip(self.parameters, gradients, strict=True)):
            assert torch.equal(p.grad.cpu(), state[f"clipped-joint-gradient/{i}"])
            state[f"correction-gradient/{i}"] = (g.detach() / self.protocol["loss_scale"]).cpu()
        del logits, loss, gradients
        self.last_step_state = state
        self.last_step = {
            "step": step,
            "example_indices": row["example_indices"],
            "pre_update_logged_task_nll": row["task_nll"],
            "pre_update_logged_preserve_nll": row["preserve_nll"],
            "pre_update_grad_task_nll": float(token_losses(before_grad).mean()),
            "pre_update_no_grad_task_nll": float(token_losses(before_no_grad).mean()),
            "pre_update_mode_comparison": mode_comparison(before_grad, before_no_grad),
            "parameter_names": self.parameter_names,
            "optimizer_group": {
                k: optimizer.param_groups[0][k] for k in ("lr", "betas", "eps", "weight_decay")
            },
        }
        _sync(self.model)
        self.pre_diagnostic_seconds = time.monotonic() - started
        self.diagnostic_seconds += self.pre_diagnostic_seconds

    def after_update(self, step, row, optimizer):
        _sync(self.model)
        arrived = time.monotonic()
        self.training_seconds += arrived - self.last_exit - self.pre_diagnostic_seconds
        if step not in self.protocol["checkpoint_steps"]:
            self.last_exit = time.monotonic()
            return
        self.check_time()
        started = time.monotonic()
        values = teacher(self.model, self.measured)
        measured = calibration(values, len(self.task), self.protocol["gate_probability"])
        filename = f"{self.prefix}-step-{step}-calibration.safetensors"
        save_file(values, self.output / filename)
        self.monitoring_seconds += time.monotonic() - started
        snapshot = {
            "step": step,
            "calibration": measured,
            "teacher_file": filename,
            "training_seconds": self.training_seconds,
            "monitoring_seconds": self.monitoring_seconds,
        }
        first = measured["adequate"] and self.selected_step is None
        if first:
            self.selected_step = step
            started = time.monotonic()
            self.save_patch(self.output / f"{self.prefix}-early-adapter", step)
            snapshot["serialization_seconds"] = time.monotonic() - started
            self.serialization_seconds += snapshot["serialization_seconds"]
        if first or step == self.protocol["steps"]:
            started = time.monotonic()
            with_grad = grad_teacher(self.model, self.measured)
            save_file(
                with_grad, self.output / f"{self.prefix}-step-{step}-grad-calibration.safetensors"
            )
            snapshot["mode_comparison"] = mode_comparison(with_grad, values)
            self.diagnostic_seconds += time.monotonic() - started
        if self.diagnostic and step == self.protocol["steps"]:
            started = time.monotonic()
            index = row["example_indices"][0]
            no_grad_chunks = token_losses(values).split(values["lengths"].tolist())
            grad_chunks = token_losses(with_grad).split(with_grad["lengths"].tolist())
            self.last_step.update(
                post_update_no_grad_task_nll=float(no_grad_chunks[index].mean()),
                post_update_grad_task_nll=float(grad_chunks[index].mean()),
            )
            joint_dot = task_dot = update_sq = joint_sq = task_sq = 0.0
            state = self.last_step_state
            for i, p in enumerate(self.parameters):
                after = p.detach().cpu().clone()
                state[f"parameter-after/{i}"] = after
                update = after.double() - state[f"parameter-before/{i}"].double()
                joint = state[f"clipped-joint-gradient/{i}"].double()
                task = state[f"correction-gradient/{i}"].double()
                joint_dot += float((joint * update).sum())
                task_dot += float((task * update).sum())
                update_sq += float(update.square().sum())
                joint_sq += float(joint.square().sum())
                task_sq += float(task.square().sum())
            save_file(state, self.output / f"{self.prefix}-final-update.safetensors")
            self.last_step.update(
                clipped_joint_gradient_dot_update=joint_dot,
                correction_gradient_dot_update=task_dot,
                update_l2=math.sqrt(update_sq),
                clipped_joint_gradient_l2=math.sqrt(joint_sq),
                correction_gradient_l2=math.sqrt(task_sq),
            )
            write(self.output / f"{self.prefix}-final-update.json", self.last_step)
            self.last_step_state = None
            self.diagnostic_seconds += time.monotonic() - started
        self.snapshots.append(snapshot)
        write(self.output / f"{self.prefix}-checkpoints.json", self.snapshots)
        print("CHECKPOINT", self.prefix, step, measured, "selected", self.selected_step, flush=True)
        self.last_exit = time.monotonic()
