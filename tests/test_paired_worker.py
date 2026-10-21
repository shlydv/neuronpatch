import hashlib
import json

import pytest
import torch

from experiments.paired_mlp.prepare import protocol
from experiments.paired_mlp.run import BANKS, execute
from experiments.paired_mlp.verify import verify
from neuronpatch.patch import fingerprint


class TinyTokenizer:
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def __init__(self):
        self.words = {}

    def encode(self, text):
        result = []
        for word in text.split():
            if word not in self.words:
                self.words[word] = len(self.words) + 3
            result.append(self.words[word])
        return result

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert not return_dict
        rows = messages if add_generation_prompt else messages[:-1]
        ids = [1, *self.encode(rows[-1]["content"])]
        return ids if add_generation_prompt else [*ids, *self.encode(messages[-1]["content"]), 2]

    def decode(self, ids, *, skip_special_tokens):
        mapping = {v: k for k, v in self.words.items()}
        return " ".join(
            mapping.get(i, f"v{i}")
            for i in ids
            if not skip_special_tokens or i not in self.all_special_ids
        )


def tiny_data():
    cases = []
    for c, target in enumerate(("blue", "green")):
        case = {"case": c, "target": target}
        for bank in BANKS:
            count = 3 if bank == "train" else 1
            case[bank] = [
                {
                    "id": f"{c}-{bank}-{i}",
                    "prompt": f"value c{c} {bank} n{i}",
                    "answers": [target if bank.startswith("target") or bank == "train" else "red"],
                }
                for i in range(count)
            ]
        cases.append(case)
    return {"cases": cases, "scope": "CPU fixture, no language-quality claim"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_actual_instruction_worker_all_families_rates_seeds_roundtrip_and_restore(tmp_path, dtype):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(54)
    model = transformers.Qwen3ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    original = fingerprint(model)
    data = tiny_data()
    spec = protocol(hashlib.sha256((json.dumps(data, indent=2) + "\n").encode()).hexdigest())
    spec.update(
        id="cpu-fixture",
        architecture={"layers": 2, "hidden": 16, "intermediate": 32},
        minimum_base_correct_utility_test=0,
        steps=2,
        max_new_tokens=2,
    )
    tokenizer = TinyTokenizer()
    rows = execute(model, tokenizer, data, spec, tmp_path / "run", lambda: None)
    assert len(rows) == 24
    assert {r["trainable_scalars"] for r in rows} == {288}
    assert all(r["base_restored"] and r["reload_teacher_exact"] for r in rows)
    assert all(len(r["history"]) == 2 for r in rows)
    assert fingerprint(model) == original
    verified = verify(tmp_path / "run", spec, tokenizer)
    assert verified["fits"] == verified["teacher_records_recomputed"] == 24
    assert verified["decoded"] == 286
    path = tmp_path / "run/fits.json"
    fits = json.loads(path.read_text())
    fits[0]["metrics"]["task_mean_nll"] += 0.01
    path.write_text(json.dumps(fits))
    with pytest.raises(AssertionError):
        verify(tmp_path / "run", spec, tokenizer)
