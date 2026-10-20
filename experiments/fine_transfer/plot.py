"""Plot recorded training losses; these are not held-out answer accuracies."""

import argparse
import json
import math
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fits = json.loads(args.fits.read_text())
    labels = {
        "fine_augmented": ("FiNE supports, same 3 phrases", "#54749d"),
        "gradient_single": ("Single-prompt gradient supports", "#d59132"),
        "gradient_consensus": ("Cross-prompt gradient supports", "#138a73"),
        "consensus_layer_matched_random": ("Matched random supports", "#9b677f"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for case, ax in enumerate(axes):
        for row in fits:
            if row["case"] != case or row["condition"] not in labels:
                continue
            label, color = labels[row["condition"]]
            ax.plot(
                [r["step"] for r in row["history"]],
                [r["nll"] for r in row["history"]],
                label=label,
                color=color,
                linewidth=2.1,
            )
        ax.axhline(-math.log(0.9), color="#666666", linestyle=":", linewidth=1)
        ax.set_title(f"Exposed development case {case}", loc="left", fontsize=11)
        ax.set_xlabel("Recorded loss evaluation (before update)")
        ax.set_yscale("log")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Worst training token / phrase NLL")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Better training fit; transfer and preservation measured separately", fontsize=12)
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
