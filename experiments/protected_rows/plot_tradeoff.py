"""Plot the frozen pilot's observed correction/preservation trade-off."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.summary.read_text())["metrics"]
    base = metrics["base"]
    denominator = base["supported"]["base_correct"]
    missing_n = base["missing"]["n"]
    baseline_missing = round(base["missing"]["em"] * missing_n)
    fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
    ax.set(xlim=(17, denominator + 1), ylim=(40, 60))
    ax.axvspan(0.95 * denominator, denominator + 1, color="#f0f4f8", zorder=0)
    ax.fill_between(
        [0.95 * denominator, denominator + 1],
        baseline_missing + 0.05 * missing_n,
        60,
        color="#d5eadc",
        zorder=0,
    )
    ax.text(37.2, 59, "Success\nregion", fontsize=10, ha="center", va="top")
    settings = [
        ("unprotected", "Selected, unprotected", "#c55a42", "o"),
        ("protected", "Selected, protected", "#2877ac", "s"),
        ("lora", "LoRA", "#925caa", "^"),
    ]
    for prefix, label, color, marker in settings:
        points = []
        for index in range(2):
            row = metrics[f"{prefix}-{index}"]
            point = (row["supported"]["retained"], round(row["missing"]["em"] * missing_n))
            points.append(point)
            ax.annotate(
                "lower rate" if index == 0 else "higher rate",
                point,
                xytext=(5, -15 if prefix == "lora" else 8),
                textcoords="offset points",
                color=color,
                fontsize=9,
            )
        ax.plot(*zip(*points), marker=marker, color=color, label=label, lw=1.5, ms=8)
    ax.scatter(denominator, baseline_missing, color="#303b48", s=65, zorder=5)
    ax.annotate(
        "Base + both random controls",
        (denominator, baseline_missing),
        xytext=(-12, -2),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=10,
    )
    ax.set_title("No setting reaches the desired correction + preservation", loc="left")
    ax.set_xlabel(f"Previously correct supported answers retained (of {denominator}) →")
    ax.set_ylabel(f"Missing-answer questions answered UNKNOWN (of {missing_n}) →")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", frameon=False)
    fig.text(
        0.01,
        -0.05,
        "Qwen3-4B · one exploratory sample/seed · frozen exact-match scoring\n"
        "Output audit separately identifies wording differences and additional semantic harms.",
        fontsize=9,
        color="#56616b",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
