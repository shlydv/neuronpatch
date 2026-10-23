"""Bundle only the inference worker and frozen math inputs for a private T4 run."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = ROOT / "experiments/relation_learning/math_development_v1"
    sources = [
        "experiments/relation_learning/math_baseline.py",
        "experiments/relation_learning/prepare_math.py",
    ]
    files = {name: (ROOT / name).read_bytes() for name in sources}
    files.update(
        {
            "data/" + name: (data / name).read_bytes()
            for name in ("inputs.json", "protocol.json", "LICENSE.gsm8k")
        }
    )
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": "128 frozen math development questions, baseline only, zero training updates",
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "math-baseline")
        .replace("transfer-results", "math-results")
        .replace("experiments.fine_transfer.run", "experiments.relation_learning.math_baseline")
        .replace("'5700'", "'3000'")
    )
    script = (
        "import base64,hashlib,json,zlib\n"
        + f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n"
        + bootstrap
    )
    compile(script, "<math-baseline>", "exec")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-math-baseline-v1-20261023",
        "title": "neuronpatch math baseline v1 20261023",
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
                "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "script_bytes": len(script.encode()),
                "bundled_files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
