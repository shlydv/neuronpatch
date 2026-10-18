"""First-token margins on the already-exposed pilot set; never selection data."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch

ROOT = Path(__file__).parent
OLD = ROOT.parent / "grounded_qa" / "results"
SYSTEM = (
    "Answer using only the supplied record. Give only the requested value. "
    "If the record does not contain the answer, reply exactly UNKNOWN."
)


def main():
    torch.set_num_threads(4)
    summary = json.loads((OLD / "summary.json").read_text())
    rows = json.loads((OLD / "selected-0.001-11-test.json").read_text())
    tok = AutoTokenizer.from_pretrained(summary["model"], revision=summary["revision"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            summary["model"],
            revision=summary["revision"],
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to("mps")
        .eval()
    )
    unknown_id = tok.encode("UNKNOWN", add_special_tokens=False)[0]
    supported = [r for r in rows if r["kind"] == "answerable"]
    for mode in ("base", "patched"):
        if mode == "patched":
            NeuronPatch.load(model, OLD / summary["chosen"]["selected"])
        with torch.inference_mode():
            for row in supported:
                text = tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("mps")
                logits = model(**inputs).logits[0, -1]
                answer_id = tok.encode(row["answer"], add_special_tokens=False)[0]
                row[mode + "_unknown_minus_answer_logit"] = (
                    logits[unknown_id] - logits[answer_id]
                ).item()
    (ROOT / "previous_diagnostic.json").write_text(json.dumps(supported, indent=2))
    for correct in (True, False):
        group = [r for r in supported if r["correct"] == correct]
        shifts = [
            r["patched_unknown_minus_answer_logit"] - r["base_unknown_minus_answer_logit"]
            for r in group
        ]
        print(
            {
                "remained_correct": correct,
                "count": len(group),
                "mean_unknown_margin_shift": sum(shifts) / len(shifts),
            }
        )


if __name__ == "__main__":
    main()
