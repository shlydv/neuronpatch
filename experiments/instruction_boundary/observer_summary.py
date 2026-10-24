"""Summarize saved optimizer forecasts; these are local NLL diagnostics only."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(root):
    root = Path(root)
    choices = json.loads((root / "choices.json").read_text())
    report = {"selected_fits": choices, "fits": {}, "files_sha256": {}}
    for fit in json.loads((root / "fits.json").read_text()):
        label = fit["id"]
        measured_path = root / f"{label}-observer.json"
        forecast_path = root / f"{label}-observer-forecasts.json"
        rows = json.loads(measured_path.read_text())
        forecasts = json.loads(forecast_path.read_text())
        if [r["step"] for r in rows] != [r["step"] for r in forecasts] or not rows:
            raise ValueError("Missing or mismatched observation steps")
        banks = sorted(rows[0]["banks"])
        for row, forecast in zip(rows, forecasts, strict=True):
            if sorted(row["banks"]) != banks or sorted(forecast["predictions"]) != banks:
                raise ValueError("Observation banks differ")
            for bank, values in row["banks"].items():
                for name in ("before_loss", "predicted_change"):
                    if values[name] != forecast["predictions"][bank][name]:
                        raise ValueError("Forecast changed after observation")
                observed = values["after_loss"] - values["before_loss"]
                error = observed - values["predicted_change"]
                if (
                    not all(
                        math.isfinite(values[k])
                        for k in ("before_loss", "after_loss", "predicted_change")
                    )
                    or values["observed_change"] != observed
                    or values["linearization_error"] != error
                ):
                    raise ValueError("Inconsistent observation")
        report["fits"][label] = {}
        for bank in banks:
            values = [r["banks"][bank] for r in rows]
            report["fits"][label][bank] = {
                "steps": [r["step"] for r in rows],
                "sign_matches": sum(
                    ((v["observed_change"] > 0) - (v["observed_change"] < 0))
                    == ((v["predicted_change"] > 0) - (v["predicted_change"] < 0))
                    for v in values
                ),
                "observations": len(values),
                "mean_absolute_prediction_error": sum(abs(v["linearization_error"]) for v in values)
                / len(values),
                "first_predicted_change": values[0]["predicted_change"],
                "first_observed_change": values[0]["observed_change"],
                "last_observed_loss": values[-1]["after_loss"],
            }
        for path in (measured_path, forecast_path):
            report["files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["scope"] = (
        "Four sampled optimizer steps on two fixed calibration documents; same-template task "
        "and clean teacher NLL. Not independent samples, answer accuracy, or transfer prediction."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.results), indent=2) + "\n")


if __name__ == "__main__":
    main()
