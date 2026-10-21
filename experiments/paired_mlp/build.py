"""Bundle one bounded private Kaggle comparison; no credentials or base weights."""

import argparse
import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

from experiments.fine_transfer.build import BOOTSTRAP, ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        *list((ROOT / "src/neuronpatch").glob("*.py")),
        *list((ROOT / "experiments/paired_mlp").glob("*.py")),
    ]
    sources.extend(
        ROOT / "experiments/fine_transfer" / name
        for name in ("build.py", "gradient.py", "gradient_controls.py", "lora_control.py")
    )
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    for name in ("inputs.json", "protocol.json"):
        files["data/" + name] = (args.data / name).read_bytes()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": (
            "One24-fit exposed-development comparison; whole answers, "
            "matched scalar budgets; fresh bank untouched"
        ),
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "paired-response")
        .replace("transfer-results", "paired-results")
        .replace("experiments.fine_transfer.run", "experiments.paired_mlp.run")
        .replace("'5700'", "'7500'")
    )
    script = (
        "import base64,hashlib,json,zlib\n"
        + f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n"
        + bootstrap
    )
    compile(script, "<paired-kaggle>", "exec")
    assert len(script.encode()) < 1_000_000
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-paired-full-response-v1-20261022",
        "title": "neuronpatch paired full response v1 20261022",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    for name, value in (("kernel-metadata.json", metadata), ("submission.json", submission)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "script_bytes": len(script.encode()),
                "bundled_files": len(files),
                "sha256": hashlib.sha256(script.encode()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
