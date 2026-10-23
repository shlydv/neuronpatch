"""Package the reviewed baseline using the existing bounded Kaggle bootstrap."""

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
    paths = list((ROOT / "src/neuronpatch").glob("*.py"))
    paths += [ROOT / "experiments/retrieval_evidence" / x for x in ("__init__.py", "run.py")]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    files.update(
        {"data/" + x: (args.data / x).read_bytes() for x in ("inputs.json", "protocol.json")}
    )
    protocol = json.loads(files["data/protocol.json"])
    assert hashlib.sha256(files["data/inputs.json"]).hexdigest() == protocol["inputs_sha256"]
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    submission = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "file_sha256": hashes,
        "scope": "110 reviewed baseline queries, no training",
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    payload = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    bootstrap = (
        BOOTSTRAP.replace("fine-transfer", "retrieval-evidence")
        .replace("transfer-results", "retrieval-results")
        .replace("experiments.fine_transfer.run", "experiments.retrieval_evidence.run")
        .replace("'5700'", "'2100'")
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
    compile(script, "<retrieval-evidence-baseline>", "exec")
    metadata = {
        "id": "shlydv/neuronpatch-retrieval-evidence-v1-20261023",
        "title": "neuronpatch retrieval evidence baseline v1 20261023",
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
        "kernel_sources": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.py").write_text(script)
    for name, value in (("submission.json", submission), ("kernel-metadata.json", metadata)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "script_bytes": len(script.encode()),
                "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
