"""Package a no-fit replay with one successful frozen parent kernel as input."""

import argparse
import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

from experiments.fine_transfer.build import BOOTSTRAP, ROOT


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    sources.extend((ROOT / "experiments/paired_endpoint").glob("*.py"))
    sources.extend(
        ROOT / "experiments/paired_mlp" / name
        for name in ("__init__.py", "data.py", "model.py", "run.py", "selection.py", "training.py")
    )
    sources.extend(
        ROOT / "experiments/fine_transfer" / n
        for n in ("build.py", "gradient.py", "gradient_controls.py", "lora_control.py")
    )
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    files["data/protocol.json"] = (args.data / "protocol.json").read_bytes()
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": "Eight frozen endpoints,live/merged replay,zero fits,no fresh questions",
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "paired-endpoint")
        .replace("transfer-results", "endpoint-results")
        .replace("experiments.fine_transfer.run", "experiments.paired_endpoint.run")
        .replace("'5700'", "'3000'")
    )
    script = (
        "import base64,hashlib,json,zlib\n"
        + f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n"
        + bootstrap
    )
    compile(script, "<endpoint-kaggle>", "exec")
    assert len(script.encode()) < 1_000_000
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-paired-endpoint-v1-20261022",
        "title": "neuronpatch paired endpoint v1 20261022",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [protocol["parent_kernel"]],
        "model_sources": [],
    }
    for name, data in (("kernel-metadata.json", metadata), ("submission.json", submission)):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "bytes": len(script.encode()),
                "files": len(files),
                "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
