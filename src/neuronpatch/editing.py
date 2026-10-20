"""A single-device GPT-J workflow for the reference knowledge-editing recipe.

Uses ordinary linear pre-hooks; it does not require FiNE's modified model class.
This is an experimental baseline workflow, not a new localization method.
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import nn

from .contribution import contribution_indices
from .knowledge import knowledge_edit_loss
from .occurrence import OccurrenceColumnEdit


@dataclass(frozen=True)
class GPTJEditConfig:
    max_steps: int = 50
    lr: float = 0.001
    neurons_per_token: int = 5
    exclude_last_layers: int = 3
    early_stop_probability: float | None = 0.9
    alpha: float = 1.0
    beta: float = 10.0
    gamma: float = 0.1
    delta_dtype: torch.dtype = torch.bfloat16
    score_dtype: torch.dtype = torch.float16
    selection: str = "contribution"
    seed: int = 42
    add_eos: bool = False

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.max_steps, self.neurons_per_token)):
            raise ValueError("max_steps and neurons_per_token must be positive integers")
        if type(self.exclude_last_layers) is not int or self.exclude_last_layers < 0:
            raise ValueError("exclude_last_layers must be nonnegative")
        for name in ("lr", "alpha", "beta", "gamma"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
                or (name == "lr" and value == 0)
            ):
                raise ValueError("finite positive lr and nonnegative coefficients required")
        p = self.early_stop_probability
        if p is not None and (
            isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or not 0 < p <= 1
        ):
            raise ValueError("early-stop probability must be in (0, 1] or None")
        if self.delta_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("unsupported delta dtype")
        if self.score_dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("unsupported score dtype")
        if self.selection not in ("contribution", "random"):
            raise ValueError("selection must be contribution or random")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.add_eos) is not bool:
            raise ValueError("add_eos must be boolean")


@dataclass(frozen=True)
class GPTJEditResult:
    edit: OccurrenceColumnEdit
    selection: torch.Tensor
    history: list[dict]
    updates: int
    stopped_early: bool
    input_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]


def _capture(model, inputs, names, target_length):
    captured, calls, handles = {}, dict.fromkeys(names, 0), []

    def capture(name):
        def hook(module, args):
            calls[name] += 1
            x = args[0]
            if x.ndim != 3 or x.shape[:2] != inputs["input_ids"].shape:
                raise ValueError("projection input must match the full single-example sequence")
            # FiNE captures FP16 activations even when model weights are FP32.
            captured[name] = x[0, -target_length - 1 : -1].detach().half().cpu()

        return hook

    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_pre_hook(capture(name)))
        with torch.no_grad():
            # FiNE's modified GPT-J casts head logits to FP32. Newer HF models may not.
            logits = model(**inputs).logits[0].float()
        if any(count != 1 for count in calls.values()):
            raise ValueError("each target must execute exactly once during capture")
        return (
            torch.stack([captured[name] for name in names], dim=1),
            logits[-target_length - 1 : -1].argmax(-1).cpu(),
            logits[: -target_length - 1].detach().clone(),
        )
    finally:
        for handle in handles:
            handle.remove()


def fit_gptj_edit(
    model: nn.Module, tokenizer, prompt: str, target: str, config: GPTJEditConfig | None = None
) -> GPTJEditResult:
    """Fit one reference-style edit without changing original weights.

    Supports materialized single-device GPT-J models with ordinary fc_out
    projections. Prepends the GPT-J EOS/BOS and one space before the desired
    target, matching upstream. No truncation, padding, chat template or implicit
    target cleaning is applied. Returns a frozen eval-mode base with scoped edit
    hooks. On failure, original training flags and requires_grad flags are restored.

    History records losses *before* each potential optimizer update. As upstream,
    early stopping checks target NLL before updating; a converged initial example
    performs zero updates. Random supports retain replacement and interleaved
    layer/feature draws, using a local CPU generator to preserve caller RNG state.
    The baseline workflow is intended to run without an outer autocast context.
    """
    config = GPTJEditConfig() if config is None else config
    if not isinstance(config, GPTJEditConfig):
        raise ValueError("config must be GPTJEditConfig")
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or not isinstance(target, str)
        or not target.strip()
    ):
        raise ValueError("prompt and target must be nonempty strings")
    if getattr(getattr(model, "config", None), "model_type", None) != "gptj":
        raise ValueError("this workflow currently supports GPT-J only")
    try:
        names = [f"transformer.h.{i}.mlp.fc_out" for i in range(len(model.transformer.h))]
        modules = [model.get_submodule(name) for name in names]
        head = model.get_output_embeddings()
    except (AttributeError, TypeError) as error:
        raise ValueError("GPT-J projection layout required") from error
    if not modules or any(type(module) is not nn.Linear for module in modules):
        raise ValueError("ordinary GPT-J fc_out projections required")
    if len({module.in_features for module in modules}) != 1 or type(head) is not nn.Linear:
        raise ValueError("uniform FFN widths and an ordinary output head required")
    eligible = len(modules) - config.exclude_last_layers
    if eligible < 1 or config.neurons_per_token > eligible * modules[0].in_features:
        raise ValueError("neuron budget must fit eligible layers")
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type == "meta":
        raise ValueError("a materialized single-device model is required")
    device = next(iter(devices))
    eos = tokenizer.eos_token_id
    if type(eos) is not int:
        raise ValueError("tokenizer must define an integer EOS token")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    labels = tokenizer.encode(" " + target, add_special_tokens=False, return_tensors="pt")
    if (
        prompt_ids.ndim != 2
        or labels.ndim != 2
        or prompt_ids.shape[0] != 1
        or labels.shape[0] != 1
        or prompt_ids.shape[1] == 0
        or labels.shape[1] == 0
    ):
        raise ValueError("tokenizer must return nonempty single-example token tensors")
    if config.add_eos:
        labels = torch.cat([labels, torch.tensor([[eos]])], dim=-1)
    ids = torch.cat([torch.tensor([[eos]]), prompt_ids, labels], dim=-1).to(device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    history, updates, stopped = [], 0, False
    try:
        model.eval()
        activations, predicted, reference = _capture(model, inputs, names, labels.shape[1])
        if config.selection == "contribution":
            supports = contribution_indices(
                activations,
                [m.weight for m in modules],
                head.weight,
                predicted,
                config.neurons_per_token,
                exclude_last_layers=config.exclude_last_layers,
                score_dtype=config.score_dtype,
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            supports = torch.empty(labels.shape[1], config.neurons_per_token, 2, dtype=torch.int64)
            for row in supports:
                for pair in row:
                    pair[0] = torch.randint(eligible, (1,), generator=generator)[0]
                    pair[1] = torch.randint(modules[0].in_features, (1,), generator=generator)[0]
        targets = [(names[layer], feature) for layer, feature in supports.reshape(-1, 2).tolist()]
        edit = OccurrenceColumnEdit(model, targets, delta_dtype=config.delta_dtype).eval()
        optimizer = torch.optim.Adam(edit.trainable_parameters(), lr=config.lr)
        for iteration in range(config.max_steps):
            logits = edit(**inputs).logits[0].float()
            losses = knowledge_edit_loss(
                logits,
                reference,
                labels[0],
                edit.delta,
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
            )
            history.append(
                {
                    "iteration": iteration + 1,
                    **{
                        name: float(getattr(losses, name).detach())
                        for name in ("total", "nll", "kl", "repetition", "norm")
                    },
                }
            )
            if config.early_stop_probability is not None and losses.nll < -math.log(
                config.early_stop_probability
            ):
                stopped = True
                break
            losses.total.backward()
            if not torch.isfinite(edit.delta.grad).all():
                raise FloatingPointError("nonfinite edit gradients")
            optimizer.step()
            optimizer.zero_grad()
            updates += 1
            if not torch.isfinite(edit.delta).all():
                raise FloatingPointError("nonfinite edit updates")
        return GPTJEditResult(
            edit,
            supports,
            history,
            updates,
            stopped,
            tuple(ids[0].cpu().tolist()),
            tuple(labels[0].tolist()),
        )
    except Exception:
        for module, training in modes:
            module.training = training
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        raise
