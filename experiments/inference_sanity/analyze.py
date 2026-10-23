"""Recompute integrity and contrasts for the frozen local diagnostic artifacts."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path(__file__).resolve().parents[2]
    r = root / "artifacts/inference_sanity_v1/kaggle_download/sanity-results/results"
    b = root / "artifacts/retrieval_evidence_v1/kaggle_download/retrieval-results/results"
    p = read(r / "protocol.json")
    rows = read(r / "inputs.json")
    pred = read(r / "predictions.json")
    identities = read(r / "identities.json")
    done = read(r / "completion.json")
    assert p == read(root / "experiments/inference_sanity/protocol_v1.json")
    assert rows == read(root / "artifacts/inference_sanity_v1/data/inputs.json")
    assert (
        done["status"] == "complete" and done["training_steps"] == 0 and done["generations"] == 36
    )
    assert list(pred) == p["modes"]
    t = AutoTokenizer.from_pretrained(p["model"], revision=p["revision"], local_files_only=True)
    logits = load_file(r / "first_logits.safetensors")
    base_logits = load_file(b / "first_logits.safetensors")
    base = {v["id"]: v for v in read(b / "predictions.json")}
    summary = {
        "generations": 36,
        "source_inputs_and_protocol_match": True,
        "canonical_weights_match": True,
        "decode_match": True,
        "first_token_disagreements": [],
        "original_mode_replay": [],
        "numerical_comparisons": [],
        "complete_counts": {},
        "generation_disagreements": {},
    }
    for mode in p["modes"]:
        assert identities[mode]["canonical_fp16_fingerprint"] == p["base_fp16_fingerprint"]
        assert [v["id"] for v in pred[mode]] == [v["id"] for v in rows]
        summary["complete_counts"][mode] = sum(v["ended"] for v in pred[mode])
        for row, out in zip(rows, pred[mode], strict=True):
            assert (
                len(
                    t.apply_chat_template(
                        row["messages"], tokenize=True, add_generation_prompt=True
                    )
                )
                == row["prompt_tokens"]
            )
            assert 0 < len(out["tokens"]) <= p["max_new_tokens"]
            assert t.decode(out["tokens"], skip_special_tokens=True) == out["text"]
            assert out["ended"] == (out["tokens"][-1] == t.eos_token_id)
            v = logits[mode + ":" + row["id"]]
            assert v.ndim == 1 and torch.isfinite(v).all()
            if v.argmax().item() != out["tokens"][0]:
                summary["first_token_disagreements"].append([mode, row["id"]])
            if mode == "fp16" and row["id"] in p["original_ids"]:
                old = base[row["id"]]
                same = out["tokens"][: len(old["tokens"])] == old["tokens"]
                summary["original_mode_replay"].append(
                    {
                        "id": row["id"],
                        "token_prefix_identical": same,
                        "logits_equal": torch.equal(v, base_logits[row["id"]]),
                    }
                )
    for a, c in [("fp16", "fp16_split"), ("fp16_split", "fp32")]:
        key = a + " vs " + c
        summary["generation_disagreements"][key] = [
            x["id"] for x, y in zip(pred[a], pred[c]) if x["tokens"] != y["tokens"]
        ]
        for row in rows:
            x = logits[a + ":" + row["id"]].double()
            y = logits[c + ":" + row["id"]].double()
            lx = x.log_softmax(-1)
            ly = y.log_softmax(-1)
            summary["numerical_comparisons"].append(
                {
                    "contrast": key,
                    "id": row["id"],
                    "argmax_equal": x.argmax().item() == y.argmax().item(),
                    "kl_left_to_right": float((lx.exp() * (lx - ly)).sum()),
                    "max_abs_logit_difference": float((x - y).abs().max()),
                }
            )
    summary["elapsed_seconds"] = done["elapsed_seconds"]
    out = root / "experiments/inference_sanity/complete_v1"
    out.mkdir(exist_ok=True)
    (out / "integrity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "numerical_comparisons"}, indent=2))
    for row in rows:
        print("\nINPUT", row["id"])
        for mode in p["modes"]:
            x = next(v for v in pred[mode] if v["id"] == row["id"])
            print(mode, repr(x["text"]), "ENDED", x["ended"])


if __name__ == "__main__":
    main()
