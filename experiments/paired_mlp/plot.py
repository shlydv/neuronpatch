"""Plot every frozen setting, separating correction from collateral changes."""

import argparse
import json
from pathlib import Path


def figure(summary, fits, identity, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    protocol = identity["protocol"]
    conditions = protocol["conditions"]
    assert len(fits) == protocol["new_fits"]
    selected = set(summary["selected_using_validation_only"].values()) - {None}
    totals = {r["banks"]["target_test"]["total"] for r in summary["cases"]}
    assert len(totals) == 1
    transfer_total = totals.pop()
    family_names = {
        "paired_mean": "Paired gradient",
        "outgoing_mean": "Outgoing gradient",
        "paired_random": "Random",
        "lora": "LoRA",
    }
    labels = []
    for condition in conditions:
        label = family_names[condition["family"]]
        if condition["family"] == "paired_random":
            label += f" seed{condition['seed']}"
        label += f"  {condition['lr']:g}"
        if condition["id"] in selected:
            label += " *"
        labels.append(label)
    cases = sorted({r["case"] for r in fits})
    assert len(cases) == 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), layout="constrained", sharey=True)
    loss_max, mention_max = 0, 0
    all_nll = []
    for offset, (case, color, marker) in enumerate(
        zip(cases, ("#5075ae", "#ae753d"), ("o", "s"), strict=True)
    ):
        records = {r["condition"]["id"]: r for r in fits if r["case"] == case}
        scores = {r["condition"]["id"]: r for r in summary["cases"] if r["case"] == case}
        ys = [i + (offset - 0.5) * 0.22 for i in range(len(conditions))]
        nll = [max(1e-9, records[c["id"]]["metrics"]["task_worst_token_nll"]) for c in conditions]
        all_nll.extend(nll)
        hits = [scores[c["id"]]["banks"]["target_test"]["hits"] for c in conditions]
        lost = [
            sum(len(scores[c["id"]]["banks"][b]["lost"]) for b in ("scope_test", "utility_test"))
            for c in conditions
        ]
        mentions = [
            sum(
                len(scores[c["id"]]["banks"][b]["new_target_mentions"])
                for b in ("scope_test", "utility_test")
            )
            for c in conditions
        ]
        loss_max = max(loss_max, max(lost))
        mention_max = max(mention_max, max(mentions))
        for ax, values in zip(axes, (nll, hits, lost, mentions), strict=True):
            ax.scatter(
                values, ys, color=color, marker=marker, s=35, label=f"Development edit {case}"
            )
    axes[0].set_xscale("log")
    if max(all_nll) / min(all_nll) < 3:
        axes[0].set_xlim(min(all_nll) / 10, max(all_nll) * 10)
    axes[0].set(title="Training target loss", xlabel="Worst token NLL (lower is better)")
    axes[1].set(
        title="Complete requested answers",
        xlabel=f"Matches /{transfer_total} wordings",
        xlim=(-0.3, transfer_total + 0.3),
        xticks=range(transfer_total + 1),
    )
    axes[2].set(
        title="Lost accepted test answers",
        xlabel="Other subjects + utility",
        xlim=(-0.3, max(2, loss_max + 0.8)),
    )
    axes[3].set(
        title="New target mentions",
        xlabel="Other subjects + utility",
        xlim=(-0.3, max(2, mention_max + 0.8)),
    )
    for ax in axes[2:]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.set(yticks=range(len(labels)), ylim=(len(labels) - 0.5, -0.5))
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].legend(fontsize=8, loc="best")
    prefix = "CPU fixture — numerical checks only. " if protocol["id"] == "cpu-fixture" else ""
    fig.suptitle(prefix + "Matched-budget complete-answer comparison", fontsize=15)
    budgets = identity["budgets"]
    fig.supxlabel(
        f"Paired/outgoing: {budgets['paired']:,} trainables; LoRA: {budgets['lora']:,}. "
        "* Rate selected using validation only; all settings shown.\n"
        "Two counterfactual development edits; correlated wordings. EOS required. "
        "Exact-score losses include valid UK aliases; see qualitative review.\n"
        "Literal target mentions miss auxiliary-label spillover. NLL floored at1e-9 for display.",
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
    if args.output.exists():
        raise FileExistsError(args.output)
    completion = json.loads((args.results / "completion.json").read_text())
    assert completion["status"] == "complete"
    figure(
        json.loads(args.summary.read_text()),
        json.loads((args.results / "fits.json").read_text()),
        json.loads((args.results / "identity.json").read_text()),
        args.output,
    )


if __name__ == "__main__":
    main()
