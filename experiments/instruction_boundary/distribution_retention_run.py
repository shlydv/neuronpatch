"""Matched-trajectory KL control using the verified diagnostic lifecycle."""

from .loss_isolation_run import execute as execute_control
from .preference_run import main


def execute(model, tokenizer, data, spec, output, check_time, *, prior=None):
    return execute_control(
        model, tokenizer, data, spec, output, check_time, prior=prior, retention=True
    )


if __name__ == "__main__":
    main(execute_fn=execute)
