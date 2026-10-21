"""Verify pinned real chat boundaries and sizes without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from neuronpatch.supervision import chat_response

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def check():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    config_path = Path(hf_hub_download(MODEL, "config.json", revision=REVISION))
    config = json.loads(config_path.read_text())
    checks = []
    for response in (
        "Paris.",
        "I do not have enough information.",
        "The revised value is 17.",
        "नमस्ते।",
    ):
        example = chat_response(
            tokenizer,
            [
                {"role": "system", "content": "Answer briefly and accurately."},
                {"role": "user", "content": "Give the requested answer."},
            ],
            response,
            max_length=128,
        )
        tail = example.input_ids[0, example.prompt_length :].tolist()
        assert tokenizer.decode(tail[:-1], skip_special_tokens=False) == response
        assert tail[-1] == tokenizer.eos_token_id
        assert example.completion_mask.sum().item() == len(tail)
        checks.append(
            {
                "response": response,
                "prompt_tokens": example.prompt_length,
                "response_tokens_including_eos": len(tail),
                "decoded_content_exact": True,
                "stop_id": tail[-1],
            }
        )
    h, m, layers = (config[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers"))
    rank_one_lora = 3 * layers * (h + m)
    per_neuron = 3 * h
    count = (rank_one_lora + per_neuron // 2) // per_neuron
    return {
        "model": MODEL,
        "revision": REVISION,
        "model_weights_downloaded": False,
        "model_inference": False,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "hidden_size": h,
        "intermediate_size": m,
        "num_hidden_layers": layers,
        "eos_token_id": tokenizer.eos_token_id,
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "budget_example_only_not_frozen_experiment": {
            "rank_one_gate_up_down_lora_scalars": rank_one_lora,
            "paired_intermediate_indices_total": count,
            "paired_scalars": count * per_neuron,
            "outgoing_only_indices_at_identical_paired_budget": 3 * count,
            "relative_count_difference_from_lora": abs(count * per_neuron / rank_one_lora - 1),
            "scope": "Count arithmetic only; no selection, training or quality claim",
        },
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
