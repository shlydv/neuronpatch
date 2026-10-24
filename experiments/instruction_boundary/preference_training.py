"""Memory-bounded SFT/DPO comparison on identical prepared response pairs.

One graph is retained at a time. DPO immediately detaches each measured score,
then recomputes each response with its detached scalar chain-rule coefficient.
This is standard DPO with extra forward compute, not an approximate/new loss.
The caller must use deterministic eval-mode inference and own adapter lifecycle.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from neuronpatch.supervision import ResponseExample, chat_response

from .preference_objective import dpo_loss, response_log_probability


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PreparedPair:
    id: str
    identity: str
    chosen: ResponseExample
    rejected: ResponseExample


def prepare_pair(tokenizer, row, *, max_length, device):
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Nonempty pair ID required")
    samples = [
        chat_response(tokenizer, row["messages"], row[k], max_length=max_length, device=device)
        for k in ("chosen", "rejected")
    ]
    c, r = samples
    if c.prompt_length != r.prompt_length or not torch.equal(
        c.input_ids[:, : c.prompt_length], r.input_ids[:, : r.prompt_length]
    ):
        raise ValueError("Preference alternatives must share the exact prompt")
    if torch.equal(c.input_ids, r.input_ids):
        raise ValueError("Preference alternatives must tokenize differently")
    identity = digest(
        {
            "row": row,
            "token_ids": [s.input_ids.tolist() for s in samples],
            "prompt_lengths": [s.prompt_length for s in samples],
        }
    )
    return PreparedPair(row["id"], identity, c, r)


def score(model, sample):
    logits = model(
        input_ids=sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
    ).logits
    return response_log_probability(logits, sample)


def cache_references(model, pairs, *, base_identity, check_time=lambda: None):
    """Score before attaching a trainable adapter; caller binds measured base identity."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Reference model must be frozen and in eval mode")
    if not isinstance(base_identity, str) or not base_identity or not pairs:
        raise ValueError("Measured base identity and nonempty pairs required")
    if len({p.id for p in pairs}) != len(pairs):
        raise ValueError("Unique pair IDs required")
    records = []
    with torch.no_grad():
        for pair in pairs:
            check_time()
            values = [float(score(model, s)) for s in (pair.chosen, pair.rejected)]
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError("Nonfinite reference score")
            records.append(
                {
                    "id": pair.id,
                    "identity": pair.identity,
                    "chosen": values[0],
                    "rejected": values[1],
                }
            )
            if len(records) % 30 == 0:
                print("REFERENCE_PAIRS", len(records), flush=True)
    return {"base_identity": base_identity, "records": records, "records_sha256": digest(records)}


def reference_lookup(pairs, cache, *, base_identity):
    if (
        cache["base_identity"] != base_identity
        or digest(cache["records"]) != cache["records_sha256"]
    ):
        raise ValueError("Reference cache base or content identity differs")
    by_id = {r["id"]: r for r in cache["records"]}
    if len(by_id) != len(cache["records"]) or len({p.id for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair/reference ID")
    for pair in pairs:
        if pair.id not in by_id or by_id[pair.id]["identity"] != pair.identity:
            raise ValueError("Reference cache does not match tokenized response pair")
        if not all(math.isfinite(by_id[pair.id][k]) for k in ("chosen", "rejected")):
            raise ValueError("Nonfinite cached reference")
    return by_id


def backward_pair(model, pair, reference, *, objective, beta, scale):
    """Accumulate scaled gradients, leaving clipping/Adam to the common loop."""
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("Deterministic eval mode required for score recomputation")
    if objective not in ("sft", "dpo") or not math.isfinite(scale) or scale <= 0:
        raise ValueError("SFT/DPO objective and positive finite scale required")
    if reference["id"] != pair.id or reference["identity"] != pair.identity:
        raise ValueError("Mismatched reference pair")
    if objective == "sft":
        chosen = score(model, pair.chosen)
        loss = -chosen / pair.chosen.completion_mask.sum()
        (loss * scale).backward()
        return {
            "loss": float(loss.detach()),
            "chosen_logp": float(chosen.detach()),
            "forward_calls": 1,
        }
    if not torch.is_grad_enabled():
        raise ValueError("Gradient-enabled execution required for training")
    # Keep gradient mode identical to recomputation: attention dispatch can differ
    # under no_grad. Each temporary model graph is discarded immediately.
    observed = [score(model, s).detach() for s in (pair.chosen, pair.rejected)]
    # Differentiate only two scalar leaves; no model graph is kept here.
    c, r = [s.clone().requires_grad_() for s in observed]
    rc, rr = [c.new_tensor(reference[k]) for k in ("chosen", "rejected")]
    loss = dpo_loss(c, r, rc, rr, beta=beta)
    coefficients = torch.autograd.grad(loss, (c, r))
    for sample, coefficient, expected in zip(
        (pair.chosen, pair.rejected), coefficients, observed, strict=True
    ):
        actual = score(model, sample)
        if not torch.allclose(actual.detach(), expected, rtol=1e-5, atol=1e-5):
            raise RuntimeError("Score changed during deterministic DPO recomputation")
        (actual * coefficient.detach() * scale).backward()
        del actual
    return {
        "loss": float(loss.detach()),
        "chosen_logp": float(observed[0]),
        "rejected_logp": float(observed[1]),
        "reference_relative_margin": float((c - r - rc + rr).detach()),
        "forward_calls": 4,
    }


def fit_preferences(
    model, pairs, preserved, cache, *, base_identity, objective, protocol, check_time=lambda: None
):
    """Shared Adam schedule, clean preservation, finite-gradient retries and cost counts.

    Successful steps use one pair and one preserved response each. Reference
    precomputation costs are separate. Retry records retain extra forward counts.
    A nonfinite attempt never calls optimizer.step or advances the sample order.
    """
    if not pairs:
        raise ValueError("Nonempty training pairs required")
    refs = reference_lookup(pairs, cache, base_identity=base_identity)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters) or not preserved:
        raise ValueError("FP32 trainable parameters and preservation bank required")
    if type(protocol["steps"]) is not int or protocol["steps"] < 1:
        raise ValueError("Positive integer steps required")
    for key in ("lr", "loss_scale", "minimum_loss_scale", "gradient_clip", "beta"):
        if not math.isfinite(protocol[key]) or protocol[key] <= 0:
            raise ValueError("Positive finite optimization settings required")
    if (
        protocol["minimum_loss_scale"] > protocol["loss_scale"]
        or not math.isfinite(protocol["preservation_weight"])
        or protocol["preservation_weight"] < 0
    ):
        raise ValueError("Invalid scale or preservation weight")
    generator = torch.Generator().manual_seed(protocol["seed"])
    orders = [
        torch.randperm(len(rows), generator=generator).tolist() for rows in (pairs, preserved)
    ]
    optimizer = torch.optim.Adam(parameters, lr=protocol["lr"])
    scale = protocol["loss_scale"]
    history = []
    for step in range(protocol["steps"]):
        indices = [order[step % len(order)] for order in orders]
        retries = []
        while True:
            check_time()
            optimizer.zero_grad(set_to_none=True)
            pair = pairs[indices[0]]
            metrics = backward_pair(
                model, pair, refs[pair.id], objective=objective, beta=protocol["beta"], scale=scale
            )
            sample = preserved[indices[1]]
            preserve_loss = -score(model, sample) / sample.completion_mask.sum()
            (preserve_loss * protocol["preservation_weight"] * scale).backward()
            if not torch.isfinite(preserve_loss):
                raise FloatingPointError("Nonfinite preservation loss")
            if any(p.grad is None for p in parameters):
                raise RuntimeError("Missing trainable gradient")
            for p in parameters:
                p.grad.div_(scale)
            if all(torch.isfinite(p.grad).all() for p in parameters):
                break
            retries.append(
                {"scale": scale, "pair_id": pair.id, "forward_calls": metrics["forward_calls"] + 1}
            )
            if scale / 2 < protocol["minimum_loss_scale"]:
                raise FloatingPointError("Nonfinite gradients at minimum scale; no Adam update")
            scale /= 2
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, protocol["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "pair_id": pair.id,
                "example_indices": indices,
                "objective": objective,
                **metrics,
                "preserve_nll": float(preserve_loss.detach()),
                "loss_scale": scale,
                "gradient_norm": float(norm),
                "gradient_retries": retries,
                "forward_calls": metrics["forward_calls"] + 1,
            }
        )
        if step == 0 or (step + 1) % 16 == 0:
            print("PREFERENCE_STEP", objective, step + 1, "loss", history[-1]["loss"], flush=True)
    return history
