"""Choose rates using validation loss only; never load generated test answers."""

import argparse
import json
import math
from pathlib import Path


def choose(validation):
    """Accept only condition metadata and validation NLL, not evaluation records."""
    winners = {}
    seen = set()
    for row in validation:
        condition = row["condition"]
        key = (condition["family"], condition["seed"])
        loss = float(row["target_val_mean_nll"])
        rate = float(condition["lr"])
        identity = (*key, rate)
        if identity in seen or not math.isfinite(loss) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Duplicate condition or invalid validation measurement")
        seen.add(identity)
        candidate = (loss, rate, condition["id"])
        if key not in winners or candidate < winners[key]:
            winners[key] = candidate
    if not winners:
        raise ValueError("No validation measurements")
    return [
        {"family": family, "seed": seed, "id": value[2], "lr": value[1], "nll": value[0]}
        for (family, seed), value in sorted(winners.items())
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fits = json.loads(args.fits.read_text())
    protocol = json.loads(args.protocol.read_text())
    expected = {c["id"]: c for c in protocol["conditions"]}
    if len(fits) != len(expected) or {r["condition"]["id"] for r in fits} != set(expected):
        raise ValueError("Incomplete comparison")
    for row in fits:
        if row["case"] != 0 or row["condition"] != expected[row["condition"]["id"]]:
            raise ValueError("Unexpected fit")
    validation = [
        {"condition": r["condition"], "target_val_mean_nll": r["metrics"]["target_val_mean_nll"]}
        for r in fits
    ]
    result = {
        "rule": "minimum validation completion NLL; lower LR breaks ties",
        "selected": choose(validation),
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
