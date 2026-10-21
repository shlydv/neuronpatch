"""Show calibration and independent subject leakage separately."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = json.loads(args.summary.read_text())["groups"]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.7, 4.9), layout="constrained")
    labels = [
        ("calibration_leaks", 6, "Training calibration subjects", "#3783a5"),
        ("evaluation_leaks", 8, "Unseen subjects", "#d88330"),
        ("exposed_scope_leaks", 16, "Previously exposed scope subjects", "#9a9ca2"),
    ]
    for offset, (key, n, label, color) in enumerate(labels):
        x = [i + (offset - 1) * 0.23 for i in range(3)]
        counts = [groups[str(w)][key] for w in (0, 1, 10)]
        bars = ax.bar(x, [100 * c / n for c in counts], width=0.21, label=label, color=color)
        ax.bar_label(bars, labels=[f"{c}/{n}" for c in counts], fontsize=9, padding=3)
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["No preservation", "Weight 1", "Weight 10\nChosen from calibration"],
        ylim=(0, 117),
        ylabel="Other-subject outputs starting with the new target (%)",
        title="Preservation helps, but does not eliminate unwanted spread",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.supxlabel(
        "Same supports and 50 updates; two exposed facts.\n"
        "All three settings get 6/6 requested prefixes on untrained wordings.",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
