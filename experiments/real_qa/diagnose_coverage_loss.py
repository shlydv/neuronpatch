"""Measure how response-token averaging weights first-token preservation failures."""

import gc
import json
from pathlib import Path

import torch
from data import MODEL, REVISION, prompt
from peft import PeftModel
from response_loss import response_logits
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch.losses import preservation_kl

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    summary = json.loads((ROOT / "coverage_512/summary.json").read_text())
    assert len(summary["runs"]) == 2
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    selected = set(summary["anchors"])
    added = set(summary["reviewed_preservation"]["added_anchor_ids"])
    rows = [
        r for r in json.loads((ROOT / "scaled/train_512.json").read_text()) if r["id"] in selected
    ]
    assert len(rows) == 115

    def load():
        return (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("mps")
            .eval()
        )

    def batch(pair):
        prefixes = [tokenizer.encode(prompt(tokenizer, r), add_special_tokens=False) for r in pair]
        answers = [
            tokenizer.encode(r["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
            for r in pair
        ]
        width = max(len(p) + len(a) for p, a in zip(prefixes, answers))
        ids, mask, labels = [], [], []
        for p, a in zip(prefixes, answers):
            pad = width - len(p) - len(a)
            ids.append([tokenizer.pad_token_id] * pad + p + a)
            mask.append([0] * pad + [1] * (len(p) + len(a)))
            labels.append([-100] * (pad + len(p)) + a)
        return {
            k: torch.tensor(v, device="mps")
            for k, v in (("input_ids", ids), ("attention_mask", mask), ("labels", labels))
        }

    reference = load()
    result = {
        "scope": "Final checkpoints; teacher-forced gold response including EOS, all 115 anchors",
        "note": (
            "Raw distribution KL, independent of repetition-penalty decoding. "
            "Not causal proof of dilution."
        ),
        "runs": {},
    }
    for folder in ("scale_512", "coverage_512"):
        for name in summary["runs"]:
            model = PeftModel.from_pretrained(load(), ROOT / folder / name).eval()
            measurements = []
            with torch.inference_mode():
                for i in range(0, len(rows), 2):
                    pair = rows[i : i + 2]
                    inputs = batch(pair)
                    edited, labels = response_logits(model, inputs)
                    original, original_labels = response_logits(reference, inputs)
                    assert torch.equal(labels, original_labels)
                    batch_means = []
                    for row, new, old, valid in zip(pair, edited, original, labels != -100):
                        lp, lq = old[valid].log_softmax(-1), new[valid].log_softmax(-1)
                        terms = (lp.exp() * (lp - lq)).sum(-1)
                        assert torch.isfinite(terms).all()
                        batch_means.append(terms.mean())
                        measurements.append(
                            {
                                "id": row["id"],
                                "group": "added" if row["id"] in added else "original",
                                "response_tokens_including_eos": len(terms),
                                "token_kl": terms.cpu().tolist(),
                                "mean_kl": terms.mean().item(),
                                "first_kl": terms[0].item(),
                                "max_kl": terms.max().item(),
                            }
                        )
                    torch.testing.assert_close(
                        torch.stack(batch_means).mean(),
                        preservation_kl(edited, original, labels != -100),
                    )
                    del edited, original, inputs
            result["runs"][f"{folder}/{name}"] = measurements
            (ROOT / "coverage_512/token_kl_diagnostic.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(folder, name, "measured", len(measurements), flush=True)
            del model
            gc.collect()
            torch.mps.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
