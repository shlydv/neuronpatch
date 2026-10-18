"""Explicit no-edit fallback for the grounded-QA experiment's validation metrics."""


def choose_preserving_candidate(base, candidates):
    """Return a candidate name or None; use validation metrics only.

    This task-specific policy requires >=10 percentage-point improvement in missing
    evidence accuracy, <=5 points supported-answer loss and <=1/16 sentinel loss.
    It is not a statistical guarantee or a universal behavior-preservation metric.
    Candidate records contain validation, sentinel and lr keys, as in the example.
    """
    eligible = []
    for name, candidate in candidates.items():
        val = candidate["validation"]
        if (
            val["answerable"] >= base["validation"]["answerable"] - 0.05 - 1e-12
            and val["missing"] >= base["validation"]["missing"] + 0.10 - 1e-12
            and candidate["sentinel"]["accuracy"] >= base["sentinel"]["accuracy"] - 1 / 16 - 1e-12
        ):
            eligible.append(name)
    return max(
        eligible,
        key=lambda n: (
            candidates[n]["validation"]["accuracy"],
            candidates[n]["validation"]["answerable"],
            -candidates[n]["lr"],
        ),
        default=None,
    )
