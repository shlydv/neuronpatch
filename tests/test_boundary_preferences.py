import copy
import json
import math

import pytest
import torch

from experiments.instruction_boundary.preference_objective import dpo_loss, response_log_probability
from experiments.instruction_boundary.preference_pairs import build_pairs
from experiments.instruction_boundary.prepare import examples
from neuronpatch.supervision import ResponseExample


def pair_fixture():
    case = examples(
        {"context": "Name | Score\nAda | 7", "question": "Ada score?", "ideal": "7"}, "doc"
    )[0]
    teacher = {"id": case["id"], "text": "Seven.", "ended": True}
    return case, teacher


def test_same_payload_changes_trust_and_reverses_target_without_reference_answers():
    case, teacher = pair_fixture()
    templates = {"plain": "Reply only {marker}.", "request": "Please return {marker}."}
    data = build_pairs([case], [teacher], templates, allowed_documents=["doc"])
    assert len(data["pairs"]) == 4
    for inside, outside in zip(data["pairs"][::2], data["pairs"][1::2], strict=True):
        assert inside["chosen"] == outside["rejected"] == teacher["text"]
        assert outside["chosen"] == inside["rejected"] == case["marker"]
        _, _, encoded = inside["messages"][-1]["content"].partition("JSON string):\n")
        original, payload = json.loads(encoded).rsplit("\n\n", 1)
        _, _, encoded = outside["messages"][-1]["content"].partition("JSON string):\n")
        decoded, offset = json.JSONDecoder().raw_decode(encoded)
        assert decoded == original
        assert encoded[offset:] == "\n\n" + payload
    # References are evaluation data, not training labels.
    changed = copy.deepcopy(case)
    changed["expected"] = "Wrong reference label"
    assert build_pairs([changed], [teacher], templates, allowed_documents=["doc"]) == data
    assert data["clean"][0]["target"] == "Seven."
    data["clean"][0]["messages"][0]["content"] = "mutated"
    assert case["messages"][0]["content"] != "mutated"


def test_bad_membership_or_teacher_does_not_create_preference_labels():
    case, teacher = pair_fixture()
    template = {"plain": "Reply {marker}."}
    with pytest.raises(ValueError, match="allowed document"):
        build_pairs([case], [teacher], template, allowed_documents=["confirmation"])
    with pytest.raises(ValueError, match="one teacher"):
        build_pairs([case], [teacher, teacher], template, allowed_documents=["doc"])
    with pytest.raises(ValueError, match="distinct"):
        build_pairs(
            [case], [dict(teacher, text=case["marker"])], template, allowed_documents=["doc"]
        )
    data = build_pairs([case], [dict(teacher, ended=False)], template, allowed_documents=["doc"])
    assert data == {"pairs": [], "clean": [], "excluded_documents": ["doc"]}


def test_sequence_log_probability_counts_eos_and_excludes_prompt():
    ids = torch.tensor([[0, 0, 1, 2]])
    mask = torch.tensor([[False, False, True, True]])
    sample = ResponseExample(ids, torch.ones_like(mask), mask, 2)
    logits = torch.full((1, 4, 3), float("nan"), dtype=torch.float64)
    logits[0, 1] = torch.tensor([0.2, 0.6, 0.2]).double().log()
    logits[0, 2] = torch.tensor([0.3, 0.2, 0.5]).double().log()
    logits.requires_grad_()
    result = response_log_probability(logits, sample)
    assert result.item() == pytest.approx(math.log(0.6 * 0.5), abs=1e-7)
    result.backward()
    assert torch.count_nonzero(logits.grad[0, [0, 3]]) == 0
    torch.testing.assert_close(logits.grad[0, 1], torch.tensor([-0.2, 0.4, -0.2]).double())
    torch.testing.assert_close(logits.grad[0, 2], torch.tensor([-0.3, -0.2, 0.5]).double())


def test_reference_relative_dpo_value_and_finite_difference_gradient():
    beta = 0.2
    chosen = torch.tensor(math.log(0.5), dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor(math.log(0.1), dtype=torch.float64, requires_grad=True)
    rc, rr = (torch.tensor(math.log(x), dtype=torch.float64) for x in (0.2, 0.3))
    loss = dpo_loss(chosen, rejected, rc, rr, beta=beta)
    expected = math.log1p(math.exp(-beta * math.log((0.5 / 0.2) / (0.1 / 0.3))))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    eps = 1e-6
    numerical = (
        dpo_loss(chosen.detach() + eps, rejected.detach(), rc, rr, beta=beta)
        - dpo_loss(chosen.detach() - eps, rejected.detach(), rc, rr, beta=beta)
    ) / (2 * eps)
    assert chosen.grad.item() == pytest.approx(numerical.item(), abs=1e-9)
    assert chosen.grad < 0 < rejected.grad
    assert chosen.grad.item() == pytest.approx(-rejected.grad.item())
    assert dpo_loss(rc, rr, rc, rr).item() == pytest.approx(math.log(2))
    with pytest.raises(ValueError, match="reference"):
        dpo_loss(chosen, rejected, rc.requires_grad_(), rr)
    for beta in (0, -1, float("nan"), True):
        with pytest.raises(ValueError, match="beta"):
            dpo_loss(chosen, rejected, rc.detach(), rr, beta=beta)


def test_tiny_causal_model_adapter_gradient_matches_explicit_sequence_formula():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
    )
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=1, target_modules=["q_proj", "v_proj"]))
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    samples = []
    for ids in ([1, 4, 7, 2], [1, 4, 8, 9, 2]):
        ids = torch.tensor([ids])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, 2:] = True
        samples.append(ResponseExample(ids, torch.ones_like(mask), mask, 2))

    def score(sample, explicit=False):
        logits = model(
            sample.input_ids, attention_mask=sample.attention_mask, use_cache=False
        ).logits
        if explicit:
            logps = logits[:, :-1].float().log_softmax(-1)
            return (
                logps.gather(-1, sample.input_ids[:, 1:, None])
                .squeeze(-1)[sample.completion_mask[:, 1:]]
                .sum()
            )
        return response_log_probability(logits, sample)

    with torch.no_grad(), model.disable_adapter():
        refs = [score(s) for s in samples]
    scores = [score(s) for s in samples]
    actual = dpo_loss(*scores, *refs, beta=0.1)
    g1 = torch.autograd.grad(actual, params)
    c, r = [score(s, True) for s in samples]
    expected = torch.nn.functional.softplus(-0.1 * ((c - r) - (refs[0] - refs[1])))
    g2 = torch.autograd.grad(expected, params)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
    assert sum(float(g.abs().sum()) for g in g1) > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
