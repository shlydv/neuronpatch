"""Build explicit full-response supervision from an existing chat template."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResponseExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    prompt_length: int


def chat_response(tokenizer, messages, response, *, max_length, device="cpu"):
    """Tokenize one assistant response including its EOS, without truncation.

    Requires a template whose generation prefix is an exact token prefix of the
    completed conversation. Any mismatch fails rather than guessing a boundary.
    The final user message must be followed by a nonempty assistant response and
    the tokenizer's EOS. Template whitespace after EOS is omitted because actual
    generation stops there. Non-whitespace suffixes or an EOS inside the response
    are rejected. Tool/multimodal templates and multiple stop IDs are unsupported.

    This does not invoke a model, generate targets or label factual correctness.
    Tokenizer/template/model revisions still need to be pinned by the caller.
    """
    if (
        not isinstance(messages, (list, tuple))
        or not messages
        or any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in ("system", "user", "assistant")
            or not isinstance(m["content"], str)
            for m in messages
        )
        or messages[-1]["role"] != "user"
        or not isinstance(response, str)
        or not response.strip()
        or type(max_length) is not int
        or max_length < 2
    ):
        raise ValueError(
            "text chat ending in user, nonempty response and positive token cap required"
        )
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("tokenizer must define one integer EOS token")
    prompt = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if any(
        not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
        for ids in (prompt, full)
    ):
        raise ValueError("chat template must return nonempty integer token lists")
    if full[: len(prompt)] != prompt:
        raise ValueError("generation prefix differs from completed chat tokens")
    tail = full[len(prompt) :]
    if eos not in tail or tail.index(eos) == 0:
        raise ValueError("completed response must contain content followed by EOS")
    end = len(prompt) + tail.index(eos) + 1
    suffix = tokenizer.decode(full[end:], skip_special_tokens=False)
    if suffix.strip():
        raise ValueError("non-whitespace template suffix or EOS inside response")
    ids = full[:end]
    if len(ids) > max_length:
        raise ValueError("full response exceeds token cap; no truncation is performed")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.zeros_like(attention_mask)
    completion_mask[:, len(prompt) :] = True
    return ResponseExample(input_ids, attention_mask, completion_mask, len(prompt))
