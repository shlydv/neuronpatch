"""Show every frozen group's correction and preservation outcomes together."""

import argparse
import json
import math
from pathlib import Path


def figure(summary, protocol, output):
    import matplotlib.pyplot as plt

    names = [v["id"] for v in protocol["conditions"]]
    assert not summary.get("partial"), (
        "Complete the frozen grid before plotting aggregate denominators"
    )
    labels = [
        n.replace("full", "Full adapter")
        .replace("only-", "Only ")
        .replace("without-", "Without ")
        .replace("final-three", "final 3 layers")
        .replace("incoming", "incoming MLP")
        .replace("outgoing", "outgoing MLP")
        for n in names
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    cases = sorted({r["case"] for r in summary["cases"]})
    assert len(cases) == 2
    for i, case in enumerate(cases):
        rows = {r["condition"]: r for r in summary["cases"] if r["case"] == case}
        for ax, key in ((axes[0, 0], "nll"), (axes[0, 1], "max_answer_kl")):
            values = [max(1e-8, rows[n]["metrics"][key]) for n in names]
            ax.scatter(
                values,
                [j + (i - 0.5) * 0.22 for j in range(len(names))],
                color=("#5b76b6", "#b17840")[i],
                s=30,
                label=f"Development fact {case}",
            )
            ax.set_xscale("log")
    bound = -math.log(protocol["criteria"]["target_probability"])
    axes[0, 0].axvline(bound, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].axvline(
        protocol["criteria"]["max_answer_kl"], color="gray", linestyle="--", linewidth=1
    )
    axes[0, 0].set(title="Target-token loss (lower is better)", xlabel="Worst training NLL")
    axes[0, 1].set(
        title="Preservation on calibration (lower is better)", xlabel="Worst token KL (nats)"
    )
    axes[0, 0].legend(fontsize=8)
    for ax, key, total, title in (
        (axes[1, 0], "untrained", 6, "Requested starts on untrained wordings"),
        (axes[1, 1], "evaluation_leaks", 8, "Wrong replacement on other subjects"),
    ):
        values = [summary["groups"][n][key] for n in names]
        bars = ax.barh(
            range(len(names)),
            values,
            color=["#37958a" if n == "full" else "#6c72a6" for n in names],
        )
        ax.bar_label(bars, labels=[f"{v}/{total}" for v in values], padding=3, fontsize=8)
        ax.set(title=title, xlim=(0, total + 0.8), xticks=range(total + 1))
    for row in axes:
        for ax in row:
            ax.set(yticks=range(len(names)), yticklabels=labels, ylim=(len(names) - 0.5, -0.5))
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
    fig.suptitle(
        "Which parts of the frozen LoRA patch retain correction and preservation?", fontsize=14
    )
    fig.supxlabel(
        "No refitting or new settings. Two exposed facts; groups have unequal sizes and norms.\n"
        "Dashed lines: calibration eligibility bounds. "
        "Prefix scores are not complete-answer quality.",
        fontsize=9,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    figure(json.loads(args.summary.read_text()), json.loads(args.protocol.read_text()), args.output)


if __name__ == "__main__":
    main()
