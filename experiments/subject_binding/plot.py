"""Plot every fit's training and calibration trade-off; no test-based selection."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fits = json.loads(args.fits.read_text())
    assert len(fits) == 6
    assert {(r["case"], r["weight"]) for r in fits} == {(c, w) for c in (0, 1) for w in (0, 1, 10)}
    import matplotlib.pyplot as plt

    colors = {0: "#8a6475", 1: "#2874a6", 10: "#1f8a70"}
    threshold = -math.log(0.9)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for case in (0, 1):
        for row in [r for r in fits if r["case"] == case]:
            weight = row["weight"]
            color = colors[weight]
            x = [h["step"] - 1 for h in row["history"]]
            label = f"weight {weight:g}"
            axes[case, 0].plot(x, [h["nll"] for h in row["history"]], color=color, label=label)
            axes[case, 0].scatter(row["updates"], row["final"]["nll"], color=color, s=25)
            axes[case, 1].plot(
                x, [max(1e-8, h["mean_answer_kl"]) for h in row["history"]], color=color
            )
            axes[case, 1].scatter(
                row["updates"], max(1e-8, row["final"]["mean_answer_kl"]), color=color, s=25
            )
            axes[case, 2].scatter(
                row["final"]["nll"], row["final"]["max_answer_kl"], color=color, s=45
            )
            axes[case, 2].annotate(
                label,
                (row["final"]["nll"], row["final"]["max_answer_kl"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axes[case, 0].axhline(threshold, color="gray", ls="--", lw=1)
        axes[case, 0].set(yscale="log", xlabel="Updates already applied", ylabel="Worst target NLL")
        axes[case, 1].set(
            yscale="log", xlabel="Updates already applied", ylabel="Mean calibration KL (nats)"
        )
        axes[case, 2].axvline(threshold, color="gray", ls="--", lw=1)
        axes[case, 2].axhline(0.1, color="gray", ls="--", lw=1)
        axes[case, 2].set(
            xscale="log",
            yscale="log",
            xlabel="Final target NLL",
            ylabel="Worst calibration-token KL",
        )
        axes[case, 0].legend(fontsize=8)
        for ax in axes[case]:
            ax.grid(alpha=0.2)
    for ax, title in zip(
        axes[0],
        (
            "Learning the edit",
            "Preserving calibration predictions",
            "Calibration eligibility: lower left",
        ),
        strict=True,
    ):
        ax.set_title(title, fontsize=11)
    axes[0, 0].text(0.02, 0.05, "Citizenship edit", transform=axes[0, 0].transAxes, fontsize=9)
    axes[1, 0].text(
        0.02, 0.05, "Country-association edit", transform=axes[1, 0].transAxes, fontsize=9
    )
    fig.suptitle("Fixed neuron supports: correction and preservation", fontsize=14)
    fig.supxlabel(
        "Lines: live-hook losses before updates; dots: actual merged model. "
        "KL display floor 1e-8.\n"
        "Eligibility requires both dashed bounds in both cases; "
        "calibration is not unseen-subject evaluation.",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
