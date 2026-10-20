"""Package one bounded pilot with data, source hashes and CUDA preflight."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent

BOOTSTRAP = """import os, subprocess, sys
from pathlib import Path
root=Path('/tmp/protected-row-pilot');root.mkdir(exist_ok=True)
for name,source in sources.items():
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
for name,source in data.items():
    path=root/'pilot-data'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv'],check=True)
subprocess.run(['uv','venv','--python','3.11','/tmp/protected-env'],check=True)
python='/tmp/protected-env/bin/python'
packages=[name+'=='+version for name,version in runtime.items()]
subprocess.run(['uv','pip','install','--python',python,'--index-url',
    'https://pypi.org/simple',*packages],check=True)
env=os.environ.copy()
env['PYTHONPATH']=str(root)+os.pathsep+str(root/'src')
env['HF_HUB_DISABLE_XET']='1'
Path('/kaggle/working/submission.json').write_text(submission)
with Path('/kaggle/working/runtime_freeze.txt').open('w') as stream:
    subprocess.run(['uv','pip','freeze','--python',python],stdout=stream,check=True)
subprocess.run([python,'-u','-m','experiments.protected_rows.preflight',
    '--device','cuda','--output','/kaggle/working/preflight.json'],env=env,check=True)
subprocess.run([python,'-u','-m','experiments.protected_rows.run',
    '--data',str(root/'pilot-data'),
    '--protocol',str(root/'experiments/protected_rows/pilot1_protocol.json'),
    '--output','/kaggle/working/pilot-results'],env=env,check=True)
"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slug", default="neuronpatch-protected-rows-pilot1-20261020")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = list((PROJECT / "src/neuronpatch").glob("*.py"))
    paths += [
        ROOT / name
        for name in (
            "run.py",
            "helpers.py",
            "preflight.py",
            "prototype.py",
            "pilot1_protocol.json",
        )
    ]
    paths += [PROJECT / "experiments/real_qa" / name for name in ("data.py", "response_loss.py")]
    sources = {str(path.relative_to(PROJECT)): path.read_text() for path in paths}
    data = {path.name: path.read_text() for path in args.data.glob("*.json")}
    protocol = json.loads((ROOT / "pilot1_protocol.json").read_text())
    if (
        hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
        != protocol["data_manifest_sha256"]
    ):
        raise ValueError("Data/protocol mismatch")
    revision = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()
    submission = json.dumps(
        {
            "kernel": f"shlydv/{args.slug}",
            "revision_at_packaging": revision,
            "source_sha256": {
                name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()
            },
            "data_sha256": {
                name: hashlib.sha256(value.encode()).hexdigest() for name, value in data.items()
            },
            "scope": "bounded exploratory model comparison; source hashes identify executed code",
        },
        indent=2,
    )
    script = (
        f"sources={sources!r}\ndata={data!r}\nruntime={protocol['runtime']!r}\n"
        f"submission={submission!r}\n" + BOOTSTRAP
    )
    compile(script, "<protected-row-kaggle-pilot>", "exec")
    (args.output / "run.py").write_text(script)
    (args.output / "submission.json").write_text(submission + "\n")
    metadata = {
        "id": f"shlydv/{args.slug}",
        "title": args.slug.replace("-", " "),
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
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
