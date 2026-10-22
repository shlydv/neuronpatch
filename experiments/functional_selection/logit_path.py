"""Separate softmax curvature from finite parameter-to-logit path changes.

This post-hoc audit uses saved logits only. The quadratic form is evaluated on
the observed logit displacement, not on an unmeasured exact Jacobian product.
It therefore cannot by itself distinguish sketch noise, network curvature and
parameter rounding. No intervention, criterion or model execution is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def moments(base, delta):
    """Per-position exact forward KL and base-softmax quadratic displacement."""
    if base.ndim != 2 or base.shape != delta.shape or not len(base):
        raise ValueError("matching nonempty [positions,vocabulary] arrays required")
    base, delta = base.double(), delta.double()
    if not torch.isfinite(base).all() or not torch.isfinite(delta).all():
        raise ValueError("finite logits and displacements required")
    logp = base.log_softmax(-1)
    p = logp.exp()
    mean = (p * delta).sum(-1)
    kl = torch.logsumexp(logp + delta, -1) - mean
    quadratic = 0.5 * (p * (delta - mean[:, None]).square()).sum(-1)
    if float(kl.min()) < -1e-10:
        raise FloatingPointError("negative KL beyond numerical tolerance")
    return kl.clamp_min(0), quadratic


def compare(base, small, large, lengths, factor):
    """Equal-example summaries for two observed scales along one weight direction."""
    if factor <= 1 or base.shape != small.shape or base.shape != large.shape:
        raise ValueError("matching logits and a scale ratio greater than one required")
    if any(n <= 0 for n in lengths) or sum(lengths) != len(base):
        raise ValueError("positive response lengths must partition logits")
    rows = []
    offset = 0
    for length in lengths:
        z = base[offset : offset + length].double()
        ds = small[offset : offset + length].double() - z
        dl = large[offset : offset + length].double() - z
        ks, qs = moments(z, ds)
        kl, ql = moments(z, dl)
        extrapolated, _ = moments(z, factor * ds)
        interpolated, _ = moments(z, dl / factor)
        _, bend = moments(z, dl / factor - ds)
        rows.append(
            {
                "small_kl": float(ks.mean()),
                "small_finite_logit_quadratic": float(qs.mean()),
                "large_kl": float(kl.mean()),
                "large_finite_logit_quadratic": float(ql.mean()),
                "large_kl_from_small_logit_secant": float(extrapolated.mean()),
                "small_kl_from_large_logit_secant": float(interpolated.mean()),
                "logit_bend_base_quadratic": float(bend.mean()),
            }
        )
        offset += length
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def audit(results, proof):
    def checked(name):
        path = results / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == proof["file_sha256"][name]
        return path

    protocol = json.loads(checked("protocol.json").read_text())
    records = json.loads(checked("records.json").read_text())
    low, high = protocol["scales"]
    rows = []
    for case in sorted({r["case"] for r in records}):
        base = load_file(str(checked(f"case-{case}-base-teacher.safetensors")))
        lengths = base["lengths"].tolist()
        for condition in protocol["conditions"]:
            pair = [r for r in records if r["case"] == case and r["condition"] == condition]
            small, large = [next(r for r in pair if r["scale"] == s) for s in (low, high)]
            tensors = [
                load_file(str(checked(f"{r['prefix']}-teacher.safetensors")))
                for r in (small, large)
            ]
            assert all(torch.equal(t["lengths"], base["lengths"]) for t in tensors)
            assert all(torch.equal(t["labels"], base["labels"]) for t in tensors)
            # The frozen protocol supplies three correction and six calibration
            # responses. Remaining scope/utility responses are held out of scoring.
            for bank, start, end in (
                ("task", 0, 3),
                ("calibration", 3, 9),
                ("heldout", 9, len(lengths)),
            ):
                first, last = sum(lengths[:start]), sum(lengths[:end])
                values = compare(
                    base["logits"][first:last],
                    tensors[0]["logits"][first:last],
                    tensors[1]["logits"][first:last],
                    lengths[start:end],
                    high / low,
                )
                row = {"case": case, "condition": condition, "bank": bank, **values}
                if bank == "calibration":
                    row["small_sketch_prediction"] = small["predicted_calibration_mean_anchored_kl"]
                    row["large_sketch_prediction"] = large["predicted_calibration_mean_anchored_kl"]
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("results", "integrity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.integrity.read_text())
    assert proof["complete"]
    result = {
        "post_hoc": True,
        "new_model_executions": 0,
        "frozen_criteria_changed": False,
        "integrity_sha256": hashlib.sha256(args.integrity.read_bytes()).hexdigest(),
        "rows": audit(args.results, proof),
        "limits": __doc__,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
