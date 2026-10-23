"""Post-hoc greedy-trie margins from verified saved prefix logits, without model calls.

This does not change the frozen0.02 probability-drop qualification. It checks a
different hypothesis: whether preserving the actual greedy decisions is a more
direct protection target. Cached generation and direct-prefix execution can
differ numerically; agreement is measured rather than assumed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.paired_mlp.data import whole_hit


def greedy_trace(logits, prefixes, paths, eos_token_id):
    paths = tuple(dict.fromkeys(tuple(p) for p in paths))
    prefixes = [tuple(p) for p in prefixes]
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all()
        or len(prefixes) != len(logits)
        or len(set(prefixes)) != len(prefixes)
        or type(eos_token_id) is not int
        or not 0 <= eos_token_id < logits.shape[1]
        or not paths
        or any(len(p) < 2 or p[-1] != eos_token_id or eos_token_id in p[:-1] for p in paths)
        or any(type(t) is not int or not 0 <= t < logits.shape[1] for p in paths for t in p)
    ):
        raise ValueError("finite prefix logits and complete distinct EOS-ended paths required")
    expected = {p[:i] for p in paths for i in range(len(p))}
    if set(prefixes) != expected:
        raise ValueError("every and only accepted-path prediction prefix must be present")
    lookup = dict(zip(prefixes, logits, strict=True))
    prefix, trace = (), []
    while True:
        z = lookup[prefix].double()
        allowed = sorted({p[len(prefix)] for p in paths if p[: len(prefix)] == prefix})
        best_allowed = allowed[int(z[allowed].argmax())]
        mask = torch.ones(z.numel(), dtype=torch.bool, device=z.device)
        mask[allowed] = False
        outside = torch.where(mask)[0]
        best_outside = int(outside[z[outside].argmax()]) if len(outside) else None
        chosen = int(z.argmax())
        gap = float(z[best_allowed] - z[best_outside]) if best_outside is not None else None
        probabilities = z.softmax(-1)
        trace.append(
            {
                "prefix": list(prefix),
                "chosen": chosen,
                "best_allowed": best_allowed,
                "best_outside": best_outside,
                "allowed_minus_outside_logit_gap": gap,
                "chosen_probability": float(probabilities[chosen]),
                "best_allowed_probability": float(probabilities[best_allowed]),
                "best_outside_probability": (
                    float(probabilities[best_outside]) if best_outside is not None else None
                ),
            }
        )
        prefix += (chosen,)
        if chosen not in allowed:
            return {
                "accepted": False,
                "observed_tokens": list(prefix),
                "trace": trace,
                "complete_response": False,
            }
        if chosen == eos_token_id:
            assert prefix in paths
            return {
                "accepted": True,
                "observed_tokens": list(prefix),
                "trace": trace,
                "complete_response": True,
            }


def audit(results, integrity, eos_token_id, tokenizer=None):
    proof = json.loads(integrity.read_text())
    assert proof["complete"]
    for name, digest in proof["file_sha256"].items():
        assert hashlib.sha256((results / name).read_bytes()).hexdigest() == digest
    spec = json.loads((results / "protocol.json").read_text())
    records = json.loads((results / "records.json").read_text())
    rows = []
    for state in records:
        stem = f"case-{state['case']}-{state['method']}"
        logits = load_file(str(results / f"{stem}-contexts.safetensors"))["logits"]
        answers = json.loads((results / f"{stem}-answers.json").read_text())["probes"]
        offset = 0
        for row, answer in zip(spec["rows"][str(state["case"])], answers, strict=True):
            n = len(row["prefixes"])
            trace = greedy_trace(
                logits[offset : offset + n], row["prefixes"], row["paths"], eos_token_id
            )
            if tokenizer is not None:
                for step in trace["trace"]:
                    step["prefix_text"] = tokenizer.decode(step["prefix"])
                    step["chosen_text"] = tokenizer.decode([step["chosen"]])
                    step["best_allowed_text"] = tokenizer.decode([step["best_allowed"]])
                    step["best_outside_text"] = (
                        tokenizer.decode([step["best_outside"]])
                        if step["best_outside"] is not None
                        else None
                    )
            offset += n
            observed = trace["observed_tokens"]
            agreement = answer["tokens"][: len(observed)] == observed
            if trace["complete_response"]:
                agreement = agreement and len(answer["tokens"]) == len(observed)
            rows.append(
                {
                    "case": state["case"],
                    "method": state["method"],
                    "id": row["id"],
                    "retention_eligible": row["retention_eligible"],
                    "direct_prefix_trace": trace,
                    "cached_generation_accepted": whole_hit(answer, row["accepted_aliases"]),
                    "cached_generation_agrees_on_observed_trace": agreement,
                }
            )
        assert offset == len(logits)
    return {
        "post_hoc": True,
        "new_model_executions": 0,
        "frozen_criteria_changed": False,
        "rows": rows,
        "all_cached_traces_agree": all(
            r["cached_generation_agrees_on_observed_trace"] for r in rows
        ),
        "limits": (
            "Exact finite-trie decision diagnostic for saved no-cache conditionals; "
            "not a trained controller,semantic oracle,unseen-task guarantee,or replacement "
            "of the frozen probability qualification."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("results", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    spec = json.loads((args.results / "protocol.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    assert tokenizer.eos_token_id == args.eos_token_id
    result = audit(args.results, args.integrity, args.eos_token_id, tokenizer)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(result["rows"]),
                "all_cached_traces_agree": result["all_cached_traces_agree"],
            }
        )
    )


if __name__ == "__main__":
    main()
