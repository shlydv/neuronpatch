"""Package only the explicit worker dependencies and hash-bound private inputs."""

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
    parser.add_argument("--kernel", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/instruction_boundary/preference_protocol_v1.json",
    )
    parser.add_argument(
        "--worker",
        choices=(
            "preference_run",
            "loss_isolation_run",
            "distribution_retention_run",
            "cross_context_run",
        ),
        default="preference_run",
    )
    parser.add_argument("--teacher-bank", type=Path)
    args = parser.parse_args()
    if not args.kernel.startswith("shlydv/neuronpatch-"):
        raise ValueError("Personal neuronpatch kernel required")
    spec = json.loads(args.protocol.read_text())
    needs_prior = args.worker in ("loss_isolation_run", "distribution_retention_run")
    if needs_prior and not spec.get("prior_kernel", "").startswith("shlydv/neuronpatch-"):
        raise ValueError("Personal prior kernel required for loss isolation")
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    sources += [
        ROOT / "experiments/paired_mlp/model.py",
        ROOT / "experiments/fine_transfer/lora_control.py",
    ]
    sources += [
        ROOT / f"experiments/instruction_boundary/{n}.py"
        for n in (
            "preference_run",
            "loss_isolation_run",
            "distribution_retention_run",
            "response_retention",
            "cross_context_run",
            "cross_context_retention",
            "preference_experiment",
            "preference_training",
            "preference_objective",
            "checkpointing",
        )
    ]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in sources}
    private = ROOT / "artifacts/boundary_preference_preparation"
    for role, name in {
        "train": "train-reviewed-banks.json",
        "calibration": "calibration-reviewed-banks.json",
        "confirmation": "confirmation-reviewed.json",
    }.items():
        raw = (private / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["data_sha256"][role]:
            raise ValueError("Private input differs from frozen protocol")
        files[f"data/{role}.json"] = raw
    if args.worker == "cross_context_run":
        if args.teacher_bank is None:
            raise ValueError("Reviewed teacher bank required")
        raw = args.teacher_bank.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["data_sha256"]["teachers"]:
            raise ValueError("Reviewed teachers differ from protocol")
        files["data/teachers.json"] = raw
    elif args.teacher_bank is not None:
        raise ValueError("Unexpected teacher bank")
    files["data/protocol.json"] = (json.dumps(spec, indent=2) + "\n").encode()
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": spec.get("scope", {"new_fits": 4, "conditions": 5}),
        "provider_timeout_seconds": spec["provider_timeout_seconds"],
    }
    payload = {
        "files": {k: base64.b85encode(zlib.compress(v, 9)).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace(
            "raw=base64.b85decode(value)", "raw=zlib.decompress(base64.b85decode(value))"
        )
        .replace("fine-transfer", "boundary-preference")
        .replace("transfer-results", "boundary-preference-results")
        .replace("experiments.fine_transfer.run", f"experiments.instruction_boundary.{args.worker}")
        .replace("'5700'", repr(str(spec["worker_timeout_seconds"])))
    )
    script = (
        "import base64,hashlib,json,zlib\n"
        + f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n"
        + bootstrap
    )
    compile(script, "<boundary-preference>", "exec")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": args.kernel,
        "title": args.kernel.split("/")[-1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [spec["prior_kernel"]] if needs_prior else [],
        "model_sources": [],
    }
    for name, value in [("kernel-metadata.json", metadata), ("submission.json", submission)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {"script_sha256": hashlib.sha256(script.encode()).hexdigest(), "files": len(files)}
        )
    )


if __name__ == "__main__":
    main()
