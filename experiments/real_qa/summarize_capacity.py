"""Report every broad-LoRA checkpoint; no implicit best-run selection."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    summary = json.loads((ROOT / "capacity/summary.json").read_text())
    lines = [
        "# Broad MLP LoRA learning curve",
        "",
        "Generated from saved checkpoints; exploratory development results only.",
        "",
        "| Rate | Epoch | Parameters | Train supported EM /64 | Train correct refusals /64 | "
        "Anchors /20 | Val supported EM /16 | Val correct refusals /16 | "
        "Val refusals on supported /16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"].values():
        train, val = run["train"], run["validation"]
        lines.append(
            f"| {run['lr']} | {run['epoch']} | {run['parameters']} | "
            f"{round(train['answerable']['em'] * 64)} | {round(train['missing']['em'] * 64)} | "
            f"{round(run['anchor_em'] * 20)} | {round(val['answerable']['em'] * 16)} | "
            f"{round(val['missing']['em'] * 16)} | {round(val['answerable']['abstained'] * 16)} |"
        )
    lines += [
        "",
        "All-MLP rank-2 LoRA uses more parameters and potentially more steps than the sparse "
        "pilot. This is a learnability reference, not a matched-budget comparison. "
        "Exact match includes wording errors; inspect raw outputs before attributing gains "
        "to reasoning. Confirmation remains untouched.",
        "",
        f"Recorded checkpoints: {len(summary['runs'])}/6. Completion must also be verified "
        "from the live process; checkpoint count alone is not process status.",
    ]
    (ROOT / "capacity/LEARNING_CURVE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
