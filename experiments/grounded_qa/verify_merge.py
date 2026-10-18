import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpatch import NeuronPatch

p = Path("experiments/grounded_qa/results")
s = json.loads((p / "summary.json").read_text())
torch.set_num_threads(4)
m = (
    AutoModelForCausalLM.from_pretrained(
        s["model"], revision=s["revision"], dtype=torch.float32, attn_implementation="eager"
    )
    .to("mps")
    .eval()
)
t = AutoTokenizer.from_pretrained(s["model"], revision=s["revision"])
patch = NeuronPatch.load(m, p / s["chosen"]["selected"])
merged = patch.merged_model().eval()
x = t(
    "Record: Expedition Echo; captain: Mira. Question: What is the captain?", return_tensors="pt"
).to("mps")
with torch.no_grad():
    a = m(**x).logits
    b = merged(**x).logits
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    result = {
        "merged_max_logit_error": (a - b).abs().max().item(),
        "atol": 1e-4,
        "rtol": 1e-4,
        "patch_bytes": (p / s["chosen"]["selected"] / "patch.safetensors").stat().st_size,
    }
(p / "deployment.json").write_text(json.dumps(result, indent=2))
print(result)
