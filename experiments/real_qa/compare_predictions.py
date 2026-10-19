"""Paired article-bootstrap uncertainty for exploratory saved-output comparisons."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MEASURES = {
    "supported_em": ("answerable", "em"),
    "supported_f1": ("answerable", "f1"),
    "supported_refusals": ("answerable", "abstained"),
    "correct_missing_refusals": ("missing", "em"),
}


def compare(before, after, draws=10000, seed=20261019):
    """Resample whole articles, retaining paired before/after questions together.

    These intervals describe development-set sampling uncertainty, not uncertainty
    across model seeds, annotation mistakes, model selection or other task families.
    Each represented article must contain both supported and missing questions.
    """
    old = {r["id"]: r for r in before}
    new = {r["id"]: r for r in after}
    if len(old) != len(before) or len(new) != len(after) or set(old) != set(new):
        raise ValueError("prediction IDs must be unique and match")
    if not old or draws < 1:
        raise ValueError("nonempty predictions and positive draws required")
    for identifier in old:
        if any(
            old[identifier][k] != new[identifier][k]
            for k in ("article", "kind", "context", "question", "answers")
        ):
            raise ValueError("prediction evidence or grouping changed")
    articles = sorted({r["article"] for r in before})
    groups = {article: [r for r in before if r["article"] == article] for article in articles}
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(articles), size=(draws, len(articles)))
    result = {}
    for name, (kind, key) in MEASURES.items():
        counts, deltas = [], []
        for rows in groups.values():
            subset = [r for r in rows if r["kind"] == kind]
            if not subset:
                raise ValueError("every article must contain both question kinds")
            counts.append(len(subset))
            deltas.append(sum(float(new[r["id"]][key]) - float(r[key]) for r in subset))
        counts, deltas = np.asarray(counts), np.asarray(deltas)
        boot = deltas[sample].sum(axis=1) / counts[sample].sum(axis=1)
        result[name] = {
            "after_minus_before": float(deltas.sum() / counts.sum()),
            "percentile_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return {
        "articles": len(articles),
        "questions": len(before),
        "draws": draws,
        "seed": seed,
        "metrics": result,
    }


def main():
    base_path = ROOT / "scaled_baseline/val.json"
    base = json.loads(base_path.read_text())
    output = {
        "interpretation": "Exploratory development comparisons; not seed or selection uncertainty",
        "baseline_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "runs": {},
    }
    for size in (128, 512, 2048):
        directory = ROOT / f"scale_{size}"
        if not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        # Only checkpoint entries saved after training evaluation and adapter export.
        # A completed checkpoint does not mean the entire process has finished.
        for name in summary["runs"]:
            path = directory / f"{name}-val.json"
            output["runs"][f"{size}/{name}"] = compare(base, json.loads(path.read_text())) | {
                "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            }
    (ROOT / "scaled/uncertainty.json").write_text(json.dumps(output, indent=2) + "\n")
    for name, values in output["runs"].items():
        print(name, values["metrics"])


if __name__ == "__main__":
    main()
