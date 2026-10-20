from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuronpatch.editing import GPTJEditConfig, fit_gptj_edit


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return torch.tensor([[7, 8]]) if text.strip() == "new answer" else torch.tensor([[3, 4, 5]])


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc_out = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return x + self.dropout(self.mlp.fc_out(x.tanh()))


class TinyGPTJ(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="gptj")
        self.embedding = nn.Embedding(11, 8)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([Block() for _ in range(5)])
        self.lm_head = nn.Linear(8, 11)
        self.recur = False

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.transformer.h:
            x = layer(x)
        if self.recur:
            x = self.transformer.h[0](x)
        return SimpleNamespace(logits=self.lm_head(x))


def test_complete_workflow_alignment_freezing_and_save(tmp_path):
    torch.manual_seed(9)
    model = TinyGPTJ()
    original = deepcopy(model.state_dict())
    config = GPTJEditConfig(max_steps=2, neurons_per_token=2, early_stop_probability=None)
    result = fit_gptj_edit(model, Tokenizer(), "Subject is", "new answer", config)
    assert result.input_token_ids == (0, 3, 4, 5, 7, 8)
    assert result.target_token_ids == (7, 8)
    assert result.updates == 2 and len(result.history) == 2 and not result.stopped_early
    assert result.selection.shape == (2, 2, 2)
    assert result.edit.delta.dtype == torch.bfloat16
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[name])
    result.edit.save(tmp_path / "edit")


def test_initial_early_stop_is_checked_before_updating_and_eos_is_explicit():
    config = GPTJEditConfig(
        max_steps=3, neurons_per_token=2, early_stop_probability=1e-20, add_eos=True
    )
    result = fit_gptj_edit(TinyGPTJ(), Tokenizer(), "Subject is", "new answer", config)
    assert result.updates == 0 and len(result.history) == 1 and result.stopped_early
    assert result.target_token_ids == (7, 8, 0)
    assert torch.count_nonzero(result.edit.delta) == 0


def test_random_selection_is_reproducible_without_changing_caller_rng():
    torch.manual_seed(11)
    model = TinyGPTJ()
    replica = deepcopy(model)
    config = GPTJEditConfig(
        max_steps=1, neurons_per_token=2, selection="random", early_stop_probability=None, seed=42
    )
    rng = torch.random.get_rng_state().clone()
    first = fit_gptj_edit(model, Tokenizer(), "Subject is", "new answer", config)
    assert torch.equal(torch.random.get_rng_state(), rng)
    second = fit_gptj_edit(replica, Tokenizer(), "Subject is", "new answer", config)
    assert torch.equal(first.selection, second.selection)
    assert torch.equal(first.edit.delta, second.edit.delta)
    assert first.history == second.history


def test_ordinary_huggingface_gptj_without_modified_model_class(tmp_path):
    transformers = pytest.importorskip("transformers")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        config = transformers.GPTJConfig(
            n_layer=4,
            n_embd=16,
            n_head=2,
            n_inner=24,
            rotary_dim=8,
            n_positions=16,
            vocab_size=11,
        )
        model = transformers.GPTJForCausalLM(config).eval()
        original = deepcopy(model.state_dict())
        result = fit_gptj_edit(
            model,
            Tokenizer(),
            "Subject is",
            "new answer",
            GPTJEditConfig(max_steps=2, neurons_per_token=2, early_stop_probability=None),
        )
        assert result.updates == 2 and torch.count_nonzero(result.edit.delta) > 0
        assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, original[name])
        result.edit.save(tmp_path / "edit")
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("cause", ["loss", "capture"])
def test_failure_restores_flags_and_removes_hooks(monkeypatch, cause):
    model = TinyGPTJ()
    model.transformer.h[0].eval()
    model.embedding.weight.requires_grad_(False)
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    if cause == "loss":

        def fail(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr("neuronpatch.editing.knowledge_edit_loss", fail)
    else:
        model.recur = True
    with pytest.raises((RuntimeError, ValueError)):
        fit_gptj_edit(
            model,
            Tokenizer(),
            "Subject is",
            "new answer",
            GPTJEditConfig(max_steps=1, neurons_per_token=2),
        )
    assert all(m.training == mode for m, mode in modes)
    assert all(p.requires_grad == flag for p, flag in flags)
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": True},
        {"lr": 0},
        {"beta": -1},
        {"early_stop_probability": 0},
        {"selection": "guess"},
        {"seed": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        GPTJEditConfig(**kwargs)
