"""Plot the pilot's wanted transfer against the frozen follow-up's leakage."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot, scope = json.loads(args.pilot.read_text()), json.loads(args.scope.read_text())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    labels = {
        "fine-original": ("Original FiNE", "#94613d", (8, 7)),
        "fine_augmented": ("FiNE + three phrasings", "#db882c", (8, 7)),
        "gradient_single": ("Single-prompt gradient", "#2573a7", (8, -17)),
        "gradient_consensus": ("Cross-prompt gradient", "#7651a8", (-158, 8)),
        "consensus_layer_matched_random": ("Matched random", "#6b7280", (8, 7)),
    }
    for condition, (label, color, offset) in labels.items():
        p, s = pilot["groups"][condition], scope["groups"][condition]
        x = 100 * p["held_out_wording"] / p["held_out_wording_n"]
        y = 100 * s["new_target_leakage"] / s["probes"]
        ax.scatter(x, y, s=75, color=color, zorder=3)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set(
        xlim=(-5, 110),
        ylim=(-5, 76),
        xlabel="Wanted change: requested answer prefixes on untrained wordings (%)",
        ylabel="Unwanted spread: new target prefixes for other subjects (%)",
        title="Transfer alone did not establish a specific edit",
    )
    ax.grid(alpha=0.2)
    ax.text(
        0.99,
        0.03,
        "Better is toward the bottom right\n"
        "Two edited facts; 6 wordings and 16 other-subject probes",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
