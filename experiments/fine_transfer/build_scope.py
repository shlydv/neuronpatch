"""Bundle one private, bounded, no-training scope diagnostic."""

import argparse
import base64
import hashlib
import io
import json
import lzma
import subprocess
import tarfile
from pathlib import Path

from .build import BOOTSTRAP, ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    sources += [
        ROOT / "experiments/fine_transfer" / name
        for name in (
            "run.py",
            "fit.py",
            "gradient.py",
            "scope.py",
            "scope_analysis.py",
            "analyze.py",
        )
    ]
    sources += [
        ROOT / "experiments/fine_reference" / name
        for name in ("audit_generations.py", "analyze.py")
    ]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    files.update(
        {
            "data/" + str(p.relative_to(args.data)): p.read_bytes()
            for p in args.data.rglob("*")
            if p.is_file()
        }
    )
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    submission = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": "One frozen-patch scope diagnostic; ten patches; no training",
    }
    # Compress binary tensor files before text encoding. Encoding each file first
    # hides the BF16 byte structure and can exceed our conservative script cap.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        members = {
            **files,
            "bundle.json": json.dumps({"hashes": hashes, "submission": submission}).encode(),
        }
        for name, raw in sorted(members.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            output.addfile(entry, io.BytesIO(raw))
    packed = base64.b85encode(lzma.compress(archive.getvalue(), preset=9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer-v1", "fine-transfer-scope-v1")
        .replace("transfer-results", "scope-results")
        .replace("data/protocol.json", "data/execution.json")
        .replace("'5700'", "'2500'")
        .replace("experiments.fine_transfer.run", "experiments.fine_transfer.scope")
    )
    assert bootstrap != BOOTSTRAP and "experiments.fine_transfer.run" not in bootstrap
    script = (
        "import base64,hashlib,io,json,lzma,tarfile\n"
        "with tarfile.open(fileobj=io.BytesIO(lzma.decompress("
        f"base64.b85decode({packed!r})))) as archive:\n"
        " bundle=json.loads(archive.extractfile('bundle.json').read())\n"
        " bundle['files']={name:base64.b85encode(archive.extractfile(name).read()).decode() "
        "for name in bundle['hashes']}\n" + bootstrap
    )
    assert len(script.encode()) < 1_000_000, (
        "Kaggle script bundle exceeds the conservative size cap"
    )
    compile(script, "<fine-transfer-scope-kaggle>", "exec")
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-fine-transfer-scope-v1-20261020",
        "title": "neuronpatch fine transfer scope v1 20261020",
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
                "script_bytes": len(script),
                "files": len(files),
                "sha256": hashlib.sha256(script.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
