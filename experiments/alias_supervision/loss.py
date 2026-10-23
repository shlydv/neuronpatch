"""Differentiable finite-response marginal likelihood with shared prediction prefixes.

The canonical path supplies its prefix logits first, preserving its original
full-sequence teacher-forcing execution. Other aliases execute only if they
supply a previously unseen prediction prefix. Each shared prefix has one vector;
this defines the numerical policy instead of assuming bitwise prefix invariance
across different full-sequence lengths. No optimizer is created here.
"""

import torch

from experiments.guarded_training.response_sets import log_response_mass
from neuronpatch.losses import completion_nll


def canonical_loss(model, sample):
    z = model(
        input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
    ).logits
    return completion_nll(z, sample.input_ids, sample.completion_mask, sample.attention_mask)


def finite_response_mean_nll(model, samples, *, eos_token_id):
    """Negative log total accepted mass divided by fixed canonical token count.

    samples must be nonempty single-chat ResponseExamples with identical prompt
    tokens, on the model's device, ending with one EOS. Duplicate token paths
    count once. No length normalization per alias or frequency weighting occurs.
    A singleton set uses the exact existing canonical CE path, including FP32
    promotion of low-precision logits. Multi-alias prefix logits are offloaded
    to FP64 CPU with autograd intact; model activations remain on its device.
    This permits precise probability arithmetic but is not activation offload.
    """
    if model.training or not samples:
        raise ValueError("eval-mode model and at least one full response required")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("one nonnegative integer EOS token required")
    prompt, paths, distinct = None, [], []
    for sample in samples:
        if sample.input_ids.ndim != 2 or sample.input_ids.shape[0] != 1:
            raise ValueError("single-example response samples required")
        before = sample.input_ids[0, : sample.prompt_length].tolist()
        path = tuple(sample.input_ids[0, sample.prompt_length :].tolist())
        if (
            not before
            or (prompt is not None and before != prompt)
            or len(path) < 2
            or path[-1] != eos_token_id
            or eos_token_id in path[:-1]
        ):
            raise ValueError("identical nonempty prompt and complete EOS-ended responses required")
        if sample.completion_mask[0].tolist() != [False] * len(before) + [True] * len(path):
            raise ValueError("response mask must select the entire response including EOS")
        if not sample.attention_mask.all():
            raise ValueError("padding is unsupported in the single-response path")
        prompt = before
        if path not in paths:
            paths.append(path)
            distinct.append(sample)
    if len(paths) == 1:
        return canonical_loss(model, distinct[0])
    contexts = {}
    for sample, path in zip(distinct, paths, strict=True):
        missing = [i for i in range(len(path)) if path[:i] not in contexts]
        if not missing:
            continue
        logits = model(
            input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
        ).logits
        for i in missing:
            # The logit before target position predicts that response token.
            contexts[path[:i]] = logits[0, sample.prompt_length + i - 1].to(
                device="cpu", dtype=torch.float64
            )
    return -log_response_mass(contexts, paths, eos_token_id=eos_token_id) / len(paths[0])
