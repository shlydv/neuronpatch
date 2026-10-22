"""Bundle one frozen guarded-training pilot with explicit setup/worker deadlines."""

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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    for directory in ("functional_selection", "guarded_training"):
        sources.extend((ROOT / "experiments" / directory).glob("*.py"))
    for directory, names in (
        (
            "paired_mlp",
            ("__init__.py", "data.py", "model.py", "run.py", "selection.py", "training.py"),
        ),
        ("paired_endpoint", ("__init__.py", "prepare.py", "run.py")),
        ("fine_transfer", ("build.py", "gradient.py", "gradient_controls.py", "lora_control.py")),
    ):
        sources.extend(ROOT / "experiments" / directory / name for name in names)
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    files["data/protocol.json"] = args.protocol.read_bytes()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": (
            "Eight protected development fits;cached frozen supports;real "
            "save/reload/merge;no fresh confirmation"
        ),
    }
    bundle = {
        "files": {name: base64.b85encode(value).decode() for name, value in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    payload = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "guarded-training")
        .replace("transfer-results", "guarded-results")
        .replace("experiments.fine_transfer.run", "experiments.guarded_training.run")
        .replace("'5700'", "'3300'")
    )
    # Setup is separately bounded; the provider limit also covers setup and
    # runtime. Queue wait is outside those runtime scopes.
    bootstrap = bootstrap.replace(
        "for command in commands:\n   print",
        "for index,command in enumerate(commands):\n"
        "   if index<3:\n"
        "    command=['timeout','--signal=TERM',str((300,300,900)[index]),*command]\n"
        "   print",
    )
    assert "for index,command" in bootstrap
    script = (
        "import base64,hashlib,json,zlib\n"
        + f"bundle=json.loads(zlib.decompress(base64.b85decode({payload!r})))\n"
        + bootstrap
    )
    compile(script, "<guarded-training-kaggle>", "exec")
    assert len(script.encode()) < 1_000_000
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-guarded-training-v1-20261022",
        "title": "neuronpatch measured preservation training v1 20261022",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "kernel_sources": [protocol["parent_kernel"]],
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    for name, value in (("kernel-metadata.json", metadata), ("submission.json", submission)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "script_bytes": len(script.encode()),
                "files": len(files),
                "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
