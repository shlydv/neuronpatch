"""GPU entry point for the frozen coverage/objective comparison."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neuronpatch.patch import fingerprint
from neuronpatch.supervision import chat_response

from .checkpointing import enable_decoder_recomputation
from .preference_experiment import run_comparison, write
from .preference_training import prepare_pair, score


@torch.no_grad()
def generate(model, tokenizer, rows, spec, path, check_time):
    records = []
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    for row in rows:
        check_time()
        ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(ids) + spec["max_new_tokens"] > spec["max_context_tokens"]:
            raise ValueError("Generation would exceed frozen context cap")
        inputs = torch.tensor([ids], device=model.device)
        tokens = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=spec["max_new_tokens"],
            do_sample=False,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )[0, len(ids) :].tolist()
        records.append(
            {
                "id": row["id"],
                "prompt_ids": ids,
                "tokens": tokens,
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "ended": bool(tokens and tokens[-1] in eos),
            }
        )
        write(path, records)
        if len(records) % 30 == 0:
            print("GENERATED", str(path), len(records), flush=True)
    return {"records": len(records), "ended": sum(r["ended"] for r in records)}


def execute(model, tokenizer, data, spec, output, check_time):
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", spec)
    if fingerprint(model) != spec["base_fingerprint"]:
        raise ValueError("Loaded base weights differ from frozen identity")
    ids = [
        {t["document_id"] for t in data[k]["tasks"]}
        for k in ("train", "calibration", "confirmation")
    ]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("Document leakage across data roles")
    device = model.device

    def pairs(rows):
        return [
            prepare_pair(tokenizer, r, max_length=spec["max_sequence_tokens"], device=device)
            for r in rows
        ]

    banks = {k: pairs(data["train"][k + "_pairs"]) for k in ("narrow", "diverse")}
    calibration = pairs(data["calibration"]["diverse_pairs"])
    preserved = [
        chat_response(
            tokenizer,
            r["messages"],
            r["target"],
            max_length=spec["max_sequence_tokens"],
            device=device,
        )
        for r in data["train"]["clean"]
    ]
    enable_decoder_recomputation(model)

    def evaluate(fitted, directory):
        started = time.monotonic()
        summary = generate(
            fitted,
            tokenizer,
            data["confirmation"]["cases"],
            spec["generation"],
            directory / "confirmation.json",
            check_time,
        )
        values = []
        with torch.no_grad():
            for pair in calibration:
                check_time()
                values.append(
                    {
                        "id": pair.id,
                        "identity": pair.identity,
                        "chosen_logp": float(score(fitted, pair.chosen)),
                        "rejected_logp": float(score(fitted, pair.rejected)),
                    }
                )
        write(directory / "calibration.json", values)
        return {**summary, "evaluation_seconds": time.monotonic() - started}

    base_dir = output / "base"
    base_dir.mkdir()
    write(base_dir / "result.json", evaluate(model, base_dir))
    names = []
    for i in range(len(model.model.layers)):
        names.extend(f"model.layers.{i}.self_attn.{p}_proj" for p in ("q", "k", "v", "o"))
        names.extend(f"model.layers.{i}.mlp.{p}_proj" for p in ("gate", "up", "down"))
    model, results = run_comparison(
        model,
        banks,
        preserved,
        names=names,
        rank=spec["rank"],
        protocol=spec["training"],
        output=output / "comparison",
        evaluate=evaluate,
        check_time=check_time,
    )
    write(
        output / "complete.json",
        {
            "conditions": 1 + len(results),
            "base_restored": fingerprint(model) == spec["base_fingerprint"],
        },
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.data / "protocol.json").read_text())
    data = {}
    for role, identity in spec["data_sha256"].items():
        raw = (args.data / (role + ".json")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity:
            raise ValueError("Frozen data hash mismatch: " + role)
        data[role] = json.loads(raw)
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > spec["internal_timeout_seconds"]:
            raise TimeoutError("Frozen experiment runtime exceeded")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Full-model worker requires the authorized GPU")
    torch.set_num_threads(2)
    torch.manual_seed(spec["training"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    execute(model.eval().requires_grad_(False), tokenizer, data, spec, args.output, check_time)


if __name__ == "__main__":
    main()
