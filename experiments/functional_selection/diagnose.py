"""Post-hoc sketch stability diagnostics; never change the frozen selection or gates.

Noise splits reuse the same contexts with independent probe draws. Anchor folds
hold out one entire preservation context. Neither is a new finite model edit,
an independent task evaluation, or a substitute for measured held-out KL.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .directions import DirectionBank, rank_scores


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def summarize(gain, responses, mask):
    """Keep signed cross terms and flag zero risk instead of dividing by it."""
    selected = responses[:, mask].double()
    joint = float(selected.sum(1).square().mean())
    diagonal = float(selected.square().sum(1).mean())
    benefit = float(gain[mask].sum())
    return {
        "selected": mask.nonzero().flatten().tolist(),
        "neurons": int(mask.sum()),
        "predicted_unit_step_gain": benefit,
        "joint_curvature": joint,
        "diagonal_curvature": diagonal,
        "joint_to_diagonal": ratio(joint, diagonal),
        "predicted_gain_squared_per_joint_curvature": ratio(benefit**2, joint),
        "zero_measured_joint_curvature": joint == 0,
    }


def select(gain, norms, responses, eligible, widths, counts, damping_ratio):
    bank = DirectionBank((), {}, gain[eligible], norms[eligible], "saved_proposal")
    ranked = rank_scores(bank, responses[:, eligible], damping_ratio=damping_ratio)
    scores = torch.zeros_like(gain)
    scores[eligible] = ranked["scores"]
    mask = torch.zeros_like(eligible)
    offset = 0
    for width, count in zip(widths, counts, strict=True):
        allowed = eligible[offset : offset + width].nonzero().flatten()
        assert len(allowed) >= count
        order = torch.argsort(scores[offset + allowed], descending=True, stable=True)
        mask[offset + allowed[order[:count]]] = True
        offset += width
    assert offset == len(mask) and int(mask.sum()) == sum(counts)
    return mask


def diagnose(saved, widths, counts, *, probes_per_anchor, damping_ratio):
    responses = saved["responses"].double()
    gain, norms, eligible = (
        saved["gain"].double(),
        saved["direction_squared_norm"].double(),
        saved["eligible"],
    )
    assert responses.ndim == 2 and responses.shape[1] == len(gain)
    assert eligible.dtype == torch.bool and eligible.shape == gain.shape == norms.shape
    assert probes_per_anchor >= 2 and len(responses) % probes_per_anchor == 0
    anchors = len(responses) // probes_per_anchor
    assert anchors >= 2 and sum(widths) == len(gain)
    assert all(torch.isfinite(t).all() for t in (responses, gain, norms))
    full = select(gain, norms, responses, eligible, widths, counts, damping_ratio)
    assert torch.equal(full, saved["selected/gain_risk"])
    supports = {
        name.removeprefix("selected/"): summarize(gain, responses, mask)
        for name, mask in saved.items()
        if name.startswith("selected/")
    }

    def fold(name, train, heldout):
        mask = select(gain, norms, responses[train], eligible, widths, counts, damping_ratio)
        return {
            "name": name,
            "training_probe_rows": train,
            "heldout_probe_rows": heldout,
            "overlap_with_full_selection": int((mask & full).sum()),
            "training": summarize(gain, responses[train], mask),
            "heldout": summarize(gain, responses[heldout], mask),
            "heldout_gain_only_control": summarize(
                gain, responses[heldout], saved["selected/adam_gain"]
            ),
        }

    anchor_folds = []
    for anchor in range(anchors):
        heldout = list(range(anchor * probes_per_anchor, (anchor + 1) * probes_per_anchor))
        train = [i for i in range(len(responses)) if i not in heldout]
        anchor_folds.append(fold(f"anchor-{anchor}", train, heldout))
    noise_folds = []
    for probe in range(probes_per_anchor):
        heldout = list(range(probe, len(responses), probes_per_anchor))
        train = [i for i in range(len(responses)) if i not in heldout]
        noise_folds.append(fold(f"probe-{probe}-per-anchor", train, heldout))
    noise_supports = [set(r["training"]["selected"]) for r in noise_folds]
    return {
        "anchors": anchors,
        "probes_per_anchor": probes_per_anchor,
        "full_supports": supports,
        "leave_one_anchor_out": anchor_folds,
        "leave_one_probe_per_anchor_out": noise_folds,
        "noise_split_common_neurons": len(set.intersection(*noise_supports)),
        "noise_split_union_neurons": len(set.union(*noise_supports)),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("results", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.integrity.read_text())
    assert proof["complete"]

    def checked(path):
        raw = path.read_bytes()
        assert (
            hashlib.sha256(raw).hexdigest()
            == proof["file_sha256"][str(path.relative_to(args.results))]
        ), f"Changed verified input: {path.name}"
        return raw

    protocol = json.loads(checked(args.results / "protocol.json"))
    assert json.loads(checked(args.results / "completion.json"))["status"] == "complete"
    counts = json.loads(checked(args.results / "identity.json"))["layer_counts"]
    paths = [args.integrity, args.results / "protocol.json", args.results / "identity.json"]
    cases = []
    for path in sorted(args.results.glob("case-*-discovery.json")):
        discovery = json.loads(checked(path))
        selection = args.results / f"case-{discovery['case']}-selection.safetensors"
        checked(selection)
        paths.extend((path, selection))
        cases.append(
            {
                "case": discovery["case"],
                **diagnose(
                    load_file(str(selection)),
                    [len(g["indices"]) for g in discovery["candidates"]],
                    counts,
                    probes_per_anchor=protocol["fisher_probes_per_anchor"],
                    damping_ratio=protocol["damping_ratio"],
                ),
            }
        )
    assert len(cases) == protocol["expected_one_step_edits"] // (
        len(protocol["conditions"]) * len(protocol["scales"])
    )
    result = {
        "post_hoc_diagnostic": True,
        "frozen_criteria_or_interventions_changed": False,
        "new_model_executions": 0,
        "cases": cases,
        "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "limits": (
            "Fold supports are not applied to the model. These are noisy conditional Fisher "
            "sketches on the existing six contexts, not fresh-task preservation measurements. "
            "Selection stability alone establishes neither correctness nor safety. "
            "The frozen finite-edit analysis remains the primary decision."
        ),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
