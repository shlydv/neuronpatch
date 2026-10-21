"""Plot all registered fits and development scores without selecting from them."""

import argparse
import json
import math
from pathlib import Path

COLORS = ["#6c72a6", "#b0803c", "#37958a", "#a75d79"]
LABELS = {
    "consensus": "Selected neurons",
    "mean": "Mean-gradient neurons",
    "lora-low": "LoRA, lr .0003",
    "lora-high": "LoRA, lr .001",
}


def training(fits, protocol, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.3), layout="constrained")
    bound = -math.log(protocol["selection"]["target_probability"])
    cases = sorted({r["case"] for r in fits})
    assert len(cases) == 2
    assert len(fits) == protocol["new_fits"]
    for i, case in enumerate(cases):
        for j, condition in enumerate(protocol["conditions"]):
            row = next(r for r in fits if r["case"] == case and r["condition"] == condition["id"])
            color, label = COLORS[j], LABELS[condition["id"]]
            updates = [h["step"] - 1 for h in row["history"]]
            axes[i, 0].plot(
                updates, [max(1e-8, h["nll"]) for h in row["history"]], color=color, label=label
            )
            axes[i, 1].plot(
                updates, [max(1e-8, h["mean_answer_kl"]) for h in row["history"]], color=color
            )
            for col, key in ((0, "nll"), (1, "mean_answer_kl")):
                axes[i, col].scatter(
                    row["updates"], max(1e-8, row["final"][key]), color=color, s=25
                )
            x, y = max(1e-8, row["final"]["nll"]), max(1e-8, row["final"]["max_answer_kl"])
            axes[i, 2].scatter(x, y, color=color, s=40)
            axes[i, 2].annotate(
                label,
                (x, y),
                xytext=(5, 5 if j % 2 == 0 else -12),
                textcoords="offset points",
                fontsize=7,
            )
        axes[i, 0].axhline(bound, color="gray", ls="--", lw=1)
        axes[i, 2].axvline(bound, color="gray", ls="--", lw=1)
        axes[i, 2].axhline(protocol["selection"]["max_answer_kl"], color="gray", ls="--", lw=1)
        axes[i, 0].set(yscale="log", xlabel="Updates applied", ylabel="Worst target NLL")
        axes[i, 1].set(yscale="log", xlabel="Updates applied", ylabel="Mean calibration KL (nats)")
        axes[i, 2].set(
            xscale="log",
            yscale="log",
            xlabel="Final target NLL",
            ylabel="Worst calibration-token KL",
        )
        axes[i, 0].legend(fontsize=7)
        for ax in axes[i]:
            ax.grid(alpha=0.2)
        axes[i, 0].text(
            0.02, 0.05, f"Development fact {case}", transform=axes[i, 0].transAxes, fontsize=9
        )
    for ax, title in zip(
        axes[0],
        ("Learning the edit", "Preserving calibration", "Eligibility: lower left"),
        strict=True,
    ):
        ax.set_title(title, fontsize=11)
    fig.suptitle("Same prediction loss and examples; different trainable parameters", fontsize=14)
    fig.supxlabel(
        "Lines: live losses before updates. Dots: merged FP16 model. KL display floor 1e-8.\n"
        "Eligibility requires both bounds in both cases; "
        "two exposed facts, not fresh confirmation.",
        fontsize=9,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def behavior(summary, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout="constrained")
    names = list(LABELS)
    for i, name in enumerate(names):
        group = summary["groups"][name]
        for ax, key, denominator in ((axes[0], "untrained", 6), (axes[1], "evaluation_leaks", 8)):
            bar = ax.bar(
                i,
                100 * group[key] / denominator,
                color=COLORS[i],
                width=0.65,
                hatch=None if group["selected_from_calibration"] else "//",
            )
            ax.bar_label(bar, labels=[f"{group[key]}/{denominator}"], padding=3, fontsize=9)
    for ax in axes:
        ax.set(
            xticks=range(4),
            xticklabels=[LABELS[n].replace(", ", "\n") for n in names],
            ylim=(0, 115),
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set(title="Requested answer starts on untrained wordings", ylabel="Success (%)")
    axes[1].set(title="Wrong replacement on other subjects", ylabel="Unwanted spread (%)")
    axes[1].axhline(12.5, color="gray", ls="--", lw=1)
    fig.supxlabel(
        "Hatching: not selected by the training/calibration rule. Report every registered fit.\n"
        "Two exposed facts; prefix scores are not complete-answer quality. "
        "LoRA uses more parameters.",
        fontsize=9,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fits = json.loads((args.results / "fits.json").read_text())
    protocol = json.loads((args.results / "identity.json").read_text())["protocol"]
    training(fits, protocol, args.output / "fit.png")
    behavior(json.loads(args.summary.read_text()), args.output / "behavior.png")


if __name__ == "__main__":
    main()
