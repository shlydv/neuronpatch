"""Bundle the diagnostic; attach frozen parent outputs instead of uploading weights."""

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
    parser.add_argument("--checkpoint-dataset")
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    for directory in (
        "adapter_diagnosis",
        "adapter_comparison",
        "subject_binding",
        "fine_transfer",
    ):
        sources += list((ROOT / "experiments" / directory).glob("*.py"))
    sources += [
        ROOT / "experiments/fine_reference" / n for n in ("analyze.py", "audit_generations.py")
    ]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    files["data/protocol.json"] = (args.data / "protocol.json").read_bytes()
    resume = args.data / "resume.json"
    resume_config = json.loads(resume.read_text()) if resume.exists() else None
    if resume_config is not None and not args.checkpoint_dataset:
        raise ValueError("Attach a verified private checkpoint dataset for a timed-out run")
    if resume_config is None and args.checkpoint_dataset:
        raise ValueError("Checkpoint dataset requires a resume manifest")
    if resume_config is not None:
        files["data/resume.json"] = resume.read_bytes()
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    submission = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": (
            "Verified resume:two missing interventions plus replay controls;zero fits"
            if resume_config
            else "Frozen selected adapters;18 merged interventions, zero new fits"
        ),
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer-v1", "adapter-diagnosis-v1")
        .replace("transfer-results", "diagnosis-results")
        .replace("experiments.fine_transfer.run", "experiments.adapter_diagnosis.run")
        .replace("5700", "3900")
    )
    script = (
        "import base64,hashlib,json,zlib\n"
        f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n" + bootstrap
    )
    compile(script, "<diagnosis-kaggle>", "exec")
    assert len(script.encode()) < 1_000_000
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-frozen-adapter-diagnosis-v1-20261021",
        "title": "neuronpatch frozen adapter diagnosis v1 20261021",
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
    if resume_config is not None:
        metadata.update(
            id="shlydv/neuronpatch-frozen-adapter-finish-v1-20261021",
            title="neuronpatch frozen adapter finish v1 20261021",
            kernel_sources=[protocol["parent_kernel"]],
            dataset_sources=[args.checkpoint_dataset],
        )
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
