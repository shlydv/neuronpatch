"""Compare canonical-reference and accepted-set logit gradients on saved contexts.

This is an objective diagnostic, not a causal attribution from these logits to
an unrelated answer or neuron. The marginal loss uses the same fixed canonical
length denominator, so a singleton response set exactly recovers mean token CE.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.guarded_training.response_sets import log_response_mass


def compare(logits, prefixes, paths, eos_token_id):
    z = logits.detach().double().clone().requires_grad_()
    lookup = {tuple(p): i for i, p in enumerate(prefixes)}
    canonical = tuple(paths[0])
    positions = [lookup[canonical[:i]] for i in range(len(canonical))]
    ce = torch.nn.functional.cross_entropy(z[positions], torch.tensor(canonical, device=z.device))
    canonical_gradient = torch.autograd.grad(ce, z)[0]
    set_loss = -log_response_mass(
        {tuple(p): z[i] for i, p in enumerate(prefixes)}, paths, eos_token_id=eos_token_id
    ) / len(canonical)
    event_gradient = torch.autograd.grad(set_loss, z)[0]
    # Which nonterminal canonical prefix most strongly teaches EOS suppression?
    candidates = [i for i, token in enumerate(canonical) if token != eos_token_id]
    i = max(candidates, key=lambda j: float(canonical_gradient[positions[j], eos_token_id]))
    row = positions[i]
    return {
        "canonical_mean_token_ce": float(ce.detach()),
        "accepted_event_nll_per_canonical_token": float(set_loss.detach()),
        "canonical_logit_gradient_norm": float(canonical_gradient.norm()),
        "event_logit_gradient_norm": float(event_gradient.norm()),
        "strongest_nonterminal_eos_pressure": {
            "prefix": list(canonical[:i]),
            "canonical_next": canonical[i],
            "base_eos_probability": float(z[row].detach().softmax(-1)[eos_token_id]),
            "canonical_eos_gradient": float(canonical_gradient[row, eos_token_id]),
            "accepted_set_eos_gradient": float(event_gradient[row, eos_token_id]),
        },
    }


def audit(results, integrity, tokenizer):
    proof = json.loads(integrity.read_text())
    assert proof["complete"]
    for name, digest in proof["file_sha256"].items():
        assert hashlib.sha256((results / name).read_bytes()).hexdigest() == digest
    spec = json.loads((results / "protocol.json").read_text())
    rows = []
    for case, probes in spec["rows"].items():
        logits = load_file(str(results / f"case-{case}-base-contexts.safetensors"))["logits"]
        offset = 0
        for row in probes:
            n = len(row["prefixes"])
            value = compare(
                logits[offset : offset + n], row["prefixes"], row["paths"], tokenizer.eos_token_id
            )
            pressure = value["strongest_nonterminal_eos_pressure"]
            pressure["prefix_text"] = tokenizer.decode(pressure["prefix"])
            pressure["canonical_next_text"] = tokenizer.decode([pressure["canonical_next"]])
            rows.append(
                {
                    "case": int(case),
                    "id": row["id"],
                    "retention_eligible": row["retention_eligible"],
                    **value,
                }
            )
            offset += n
        assert offset == len(logits)
    return {
        "post_hoc": True,
        "new_model_executions": 0,
        "frozen_qualification_changed": False,
        "normalization": "Both losses divided by the same canonical response length including EOS.",
        "rows": rows,
        "limits": (
            "Exact objective/logit-gradient comparison on saved direct-prefix conditionals; "
            "not full parameter-gradient attribution or a causal explanation of the oxygen "
            "error. No trained benefit is established."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("results", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    spec = json.loads((args.results / "protocol.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    value = audit(args.results, args.integrity, tokenizer)
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"rows": len(value["rows"]), "new_model_executions": 0}))


if __name__ == "__main__":
    main()
