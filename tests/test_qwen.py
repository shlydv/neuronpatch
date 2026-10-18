"""Optional Transformers integration; no network, tokenizer, or model downloads."""

import runpy
from pathlib import Path

import pytest
import torch


def test_tiny_qwen_training_and_export():
    pytest.importorskip("transformers")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        example = Path(__file__).resolve().parents[1] / "examples" / "fit_tiny_qwen.py"
        result = runpy.run_path(str(example))["run"]()
        assert result["trainable_parameters"] == 32 * 8
        assert result["final_token_loss"] < result["initial_token_loss"]
    finally:
        torch.set_num_threads(previous)
