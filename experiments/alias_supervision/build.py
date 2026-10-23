"""Bundle one frozen eight-fit auxiliary-loss comparison on free private Kaggle."""

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
    spec = json.loads(args.protocol.read_text())
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    sources.extend((ROOT / "experiments/alias_supervision").glob("*.py"))
    for directory, names in (
        ("outcome_guard", ("__init__.py", "run.py")),
        ("guarded_training", ("__init__.py", "response_sets.py")),
        ("paired_endpoint", ("__init__.py", "prepare.py", "run.py")),
        (
            "paired_mlp",
            ("__init__.py", "data.py", "model.py", "run.py", "selection.py", "training.py"),
        ),
        ("fine_transfer", ("build.py", "gradient.py", "gradient_controls.py", "lora_control.py")),
    ):
        sources.extend(ROOT / "experiments" / directory / name for name in names)
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    files["data/protocol.json"] = args.protocol.read_bytes()
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": (
            "Two exact canonical controls plus six accepted-set fits; "
            "all supports and original schedules frozen"
        ),
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    payload = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "alias-supervision")
        .replace("transfer-results", "alias-results")
        .replace("experiments.fine_transfer.run", "experiments.alias_supervision.run")
        .replace("'5700'", "'5100'")
    )
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
    compile(script, "<alias-supervision-kaggle>", "exec")
    assert len(script.encode()) < 1_000_000
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-alias-supervision-v1-20261023",
        "title": "neuronpatch alias supervision v1 20261023",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "kernel_sources": [spec["parent_kernel"], spec["probe_kernel"]],
    }
    for name, value in (("kernel-metadata.json", metadata), ("submission.json", submission)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "script_bytes": len(script.encode()),
                "files": len(files),
                "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
