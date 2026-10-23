"""Package the reviewed transfer worker, private data and explicit runtime limits."""

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
    spec = json.loads(
        (ROOT / "experiments/instruction_boundary/transfer_protocol_v1.json").read_text()
    )
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    sources += [
        ROOT / name
        for name in (
            "experiments/paired_mlp/model.py",
            "experiments/paired_mlp/training.py",
            "experiments/fine_transfer/lora_control.py",
            "experiments/relation_learning/gradient_agreement.py",
            "experiments/relation_learning/update_prediction.py",
            *[
                f"experiments/instruction_boundary/{name}.py"
                for name in (
                    "transfer_run",
                    "transfer_data",
                    "probe_selection",
                    "row_gradients",
                    "prepare",
                    "run",
                )
            ],
        )
    ]
    files = {str(path.relative_to(ROOT)): path.read_bytes() for path in sources}
    data = ROOT / "artifacts/boundary_transfer_v1/data"
    files["data/inputs.json"] = (data / "inputs.json").read_bytes()
    files["data/protocol.json"] = (json.dumps(spec, indent=2) + "\n").encode()
    if hashlib.sha256(files["data/inputs.json"]).hexdigest() != spec["inputs_sha256"]:
        raise ValueError("Data differs from frozen protocol")
    (data / "protocol.json").write_bytes(files["data/protocol.json"])
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "limits": spec["limits"],
        "scope": "Eight fits, four calibration-selected arms; exploratory boundary transfer",
    }
    payload = {
        "files": {name: base64.b85encode(value).decode() for name, value in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "boundary-transfer")
        .replace("transfer-results", "boundary-transfer-results")
        .replace("experiments.fine_transfer.run", "experiments.instruction_boundary.transfer_run")
        .replace("'5700'", repr(str(spec["limits"]["worker_seconds"])))
    )
    script = (
        "import base64,hashlib,json,zlib\n"
        + f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n"
        + bootstrap
    )
    compile(script, "<boundary-transfer>", "exec")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-boundary-transfer-v1-20261024",
        "title": "neuronpatch boundary transfer v1 20261024",
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
            {"script_sha256": hashlib.sha256(script.encode()).hexdigest(), "files": len(files)}
        )
    )


if __name__ == "__main__":
    main()
