"""Audit cached canonical versus greedy prediction contexts without model calls."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def shared_contexts(canonical, generated):
    """Count identical causal prefixes,including the first diverging prediction.

    Each sequence includes its stop token. Position j predicts token j from
    tokens[:j],so unequal target tokens can still have the same prediction context.
    A context after the greedy stop token is not on the generated response path.
    """
    assert canonical and generated
    common = 0
    for a, b in zip(canonical, generated):
        if a != b:
            break
        common += 1
    return min(common + 1, len(canonical), len(generated))


def audit(parent, integrity):
    proof = json.loads(integrity.read_text())
    assert proof["complete"] and proof["fits"] == 24
    paths = [parent / "inputs.json", parent / "tokenization.json"]

    def checked(path):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == proof["file_sha256"][path.name]
        return json.loads(raw)

    data, tokenization = (checked(p) for p in paths)
    rows, cases = [], []
    for case in data["cases"]:
        path = parent / f"case-{case['case']}-base.json"
        base = checked(path)
        paths.append(path)
        answers = {r["id"]: r for r in base["preserve"]}
        local = []
        for row in case["preserve"]:
            encoded = tokenization[row["id"]]
            canonical = encoded["ids"][encoded["prompt_length"] :]
            generated = answers[row["id"]]
            assert generated["ended"] and not generated["truncated"]
            assert not generated["unexpected_special"]
            tokens = generated["tokens"]
            shared = shared_contexts(canonical, tokens)
            local.append(
                {
                    "case": case["case"],
                    "id": row["id"],
                    "canonical_tokens_including_stop": len(canonical),
                    "greedy_tokens_including_stop": len(tokens),
                    "identical_complete_token_sequence": canonical == tokens,
                    "canonical_contexts_on_greedy_path": shared,
                    "shared_context_fraction": shared / len(canonical),
                    "first_prediction_position_weight": 1 / len(canonical),
                }
            )
        rows.extend(local)
        cases.append(
            {
                "case": case["case"],
                "anchors": len(local),
                "identical_complete_token_sequences": sum(
                    r["identical_complete_token_sequence"] for r in local
                ),
                "canonical_prediction_contexts": sum(
                    r["canonical_tokens_including_stop"] for r in local
                ),
                "contexts_on_greedy_path": sum(
                    r["canonical_contexts_on_greedy_path"] for r in local
                ),
                "equal_anchor_shared_context_fraction": sum(
                    r["shared_context_fraction"] for r in local
                )
                / len(local),
            }
        )
    # Independent exhaustive prefix enumeration checks the causal off-by-one
    # convention;this is not a test of model outputs or semantic equivalence.
    sequences = [s for n in range(1, 5) for s in itertools.product((0, 1), repeat=n)]
    for a, b in itertools.product(sequences, repeat=2):
        assert shared_contexts(a, b) == sum(a[:j] == b[:j] for j in range(min(len(a), len(b))))
    return {
        "retrospective_context_audit": True,
        "frozen_protocol_or_interventions_changed": False,
        "new_model_forwards": 0,
        "cases": cases,
        "rows": rows,
        "prefix_enumeration_checks": len(sequences) ** 2,
        "file_sha256": {
            **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "parent_integrity": hashlib.sha256(integrity.read_bytes()).hexdigest(),
            "audit_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limits": (
            "Exact cached prefixes on the deterministic greedy path,not sequence probability "
            "mass. A different sequence can be a correct semantic alias. Conditional Fisher "
            "measurements remain valid on their stated canonical contexts. This audit establishes "
            "neither a cause of interference nor the performance of an untested alternative."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("parent", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parent, args.integrity)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("cases", "prefix_enumeration_checks")}, indent=2))


if __name__ == "__main__":
    main()
