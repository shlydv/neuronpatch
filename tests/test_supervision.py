import pytest
import torch

from neuronpatch.losses import completion_nll
from neuronpatch.supervision import chat_response


class Template:
    """A tokenizer double with a visible EOS and template newline."""

    eos_token_id = 7

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=True):
        if add_generation_prompt:
            result = [1, 2, 3]
        else:
            result = [1, 2, 3, 4, 5, 7, 8]
        # Transformers5 defaults to BatchEncoding, unlike older releases.
        return {"input_ids": result} if return_dict else result

    def decode(self, ids, *, skip_special_tokens):
        return "".join("\n" if i == 8 else "text" for i in ids)


def test_response_boundaries_include_every_answer_token_and_eos():
    example = chat_response(
        Template(), [{"role": "user", "content": "question"}], "answer", max_length=6
    )
    assert example.input_ids.tolist() == [[1, 2, 3, 4, 5, 7]]
    assert example.completion_mask.tolist() == [[False, False, False, True, True, True]]
    assert example.attention_mask.all() and example.prompt_length == 3
    logits = torch.randn(1, 6, 9, requires_grad=True)
    loss = completion_nll(
        logits, example.input_ids, example.completion_mask, example.attention_mask
    )
    loss.backward()
    assert logits.grad[0, :2].count_nonzero() == 0
    assert logits.grad[0, 2:5].abs().sum() > 0
    assert logits.grad[0, 5].count_nonzero() == 0


@pytest.mark.parametrize("problem", ["prefix", "suffix", "missing_eos", "empty", "cap"])
def test_no_silent_boundary_guess_or_answer_truncation(problem):
    class Broken(Template):
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, return_dict=True
        ):
            result = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                return_dict=return_dict,
            )
            if not add_generation_prompt:
                if problem == "prefix":
                    result[2] = 6
                elif problem == "suffix":
                    result[-1] = 6
                elif problem == "missing_eos":
                    result[5] = 6
                elif problem == "empty":
                    result = [1, 2, 3, 7, 8]
            return result

    with pytest.raises(ValueError):
        chat_response(
            Broken(),
            [{"role": "user", "content": "question"}],
            "answer",
            max_length=5 if problem == "cap" else 20,
        )
