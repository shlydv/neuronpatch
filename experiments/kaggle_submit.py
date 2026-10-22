"""Submit an explicitly bounded private experiment to the personal Kaggle account.

This sets the provider session runtime limit as well as any limits inside the
worker. It does not bound time spent waiting for allocation or prove that the
provider has enforced the requested limit. No notebook is restarted here.
"""

import argparse
import json
from pathlib import Path


def push_bounded(api, bundle, *, timeout_seconds):
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 10800:
        raise ValueError("an explicit provider runtime limit of 60–10800 seconds is required")
    bundle = Path(bundle).resolve()
    metadata = json.loads((bundle / "kernel-metadata.json").read_text())
    if not metadata.get("id", "").startswith("shlydv/neuronpatch-"):
        raise ValueError("experiment destination must be a personal shlydv/neuronpatch notebook")
    if metadata.get("is_private") is not True:
        raise ValueError("experiment notebooks must be private")
    if metadata.get("machine_shape") != "NvidiaTeslaT4":
        raise ValueError("this launcher supports only the reviewed T4 configuration")
    if api.get_config_value("username") != "shlydv":
        raise ValueError("authenticate the personal shlydv Kaggle account before submitting")
    script = (bundle / metadata["code_file"]).resolve()
    if not script.is_relative_to(bundle) or not script.is_file():
        raise ValueError("the code file must exist inside the reviewed bundle")
    return api.kernels_push(str(bundle), timeout=str(timeout_seconds))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(push_bounded(api, args.bundle, timeout_seconds=args.timeout_seconds))


if __name__ == "__main__":
    main()
