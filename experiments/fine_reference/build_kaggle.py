"""Package the reference harness for Kaggle's current Python3.13 host image."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BOOTSTRAP = """import subprocess, sys
from pathlib import Path
root=Path('/tmp/fine-reference-harness');root.mkdir(exist_ok=True)
for name,source in sources.items(): (root/name).write_text(source)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv'],check=True)
subprocess.run(['uv','venv','--python','3.11','/tmp/fine-env'],check=True)
python='/tmp/fine-env/bin/python'
subprocess.run(['uv','pip','install','--python',python,'--index-url',
    'https://pypi.org/simple','torch==2.0.1','numpy==1.26.4',
    'transformers==4.40.0','tokenizers==0.19.1','huggingface-hub==0.23.5',
    'accelerate==0.30.1','PyYAML','requests','tqdm'],check=True)
repo='/tmp/fine-upstream'
subprocess.run(['git','clone','--quiet',
    'https://github.com/opanhw/fine-unified-editing-code',repo],check=True)
subprocess.run(['git','-C',repo,'checkout','--quiet',
    'a2489cbac0dfe5a956385734065cfa041384e7d1'],check=True)
subprocess.run([python,'-u',str(root/'run.py'),'--upstream',repo,
    '--output','/kaggle/working/preflight','--preflight'],check=True)
subprocess.run([python,'-u',str(root/'run.py'),'--upstream',repo,
    '--output','/kaggle/working/results','--cases',str(cases)],check=True)
"""

PARITY_BOOTSTRAP = """import os, subprocess, sys
from pathlib import Path
root=Path('/tmp/fine-parity-harness');root.mkdir(exist_ok=True)
for name,source in sources.items():
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(source)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv'],check=True)
subprocess.run(['uv','venv','--python','3.11','/tmp/fine-env'],check=True)
python='/tmp/fine-env/bin/python'
subprocess.run(['uv','pip','install','--python',python,'--index-url',
    'https://pypi.org/simple','torch==2.4.1','numpy==1.26.4',
    'transformers==4.40.0','tokenizers==0.19.1','huggingface-hub==0.23.5',
    'accelerate==0.30.1','safetensors==0.4.5','PyYAML','requests','tqdm'],check=True)
repo='/tmp/fine-upstream'
subprocess.run(['git','clone','--quiet',
    'https://github.com/opanhw/fine-unified-editing-code',repo],check=True)
subprocess.run(['git','-C',repo,'checkout','--quiet',
    'a2489cbac0dfe5a956385734065cfa041384e7d1'],check=True)
env=os.environ.copy()
env['PYTHONPATH']=str(root/'src')
subprocess.run([python,'-u',str(root/'run.py'),'--upstream',repo,
    '--output','/kaggle/working/preflight','--preflight'],env=env,check=True)
Path('/kaggle/working/preflight/submission.json').write_text(submission)
with Path('/kaggle/working/preflight/runtime.txt').open('w') as stream:
    subprocess.run(['uv','pip','freeze','--python',python],stdout=stream,check=True)
for dtype in ('float32','float16'):
    subprocess.run([python,'-u',str(root/'check_edit.py'),'--upstream',repo,
        '--base-dtype',dtype],env=env,check=True)
subprocess.run([python,'-u',str(root/'gpu_parity.py'),'--upstream',repo,
    '--output','/kaggle/working/parity-results','--cases',str(cases),
    '--library-revision',library_revision],env=env,check=True)
"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--slug", default="neuronpatch-fine-reference-smoke-20261019")
    parser.add_argument("--mode", choices=("reference", "parity"), default="reference")
    args = parser.parse_args()
    if not 1 <= args.cases <= 839:
        parser.error("cases must be 1..839")
    if args.mode == "parity" and args.cases > 2:
        parser.error("parity mode is bounded to one or two cases")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    sources = {name: (root / name).read_text() for name in ("run.py", "upstream_metrics.py")}
    if args.mode == "reference":
        script = f"sources={sources!r}\ncases={args.cases}\n" + BOOTSTRAP
    else:
        project = root.resolve().parents[1]
        if subprocess.check_output(
            [
                "git",
                "-C",
                str(project),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src/neuronpatch",
            ],
            text=True,
        ):
            raise ValueError("Commit library changes before packaging parity provenance")
        revision = subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
        ).strip()
        for name in ("gpu_parity.py", "check_edit.py"):
            sources[name] = (root / name).read_text()
        for path in sorted((project / "src/neuronpatch").rglob("*.py")):
            sources[str(path.relative_to(project))] = path.read_text()
        submission = json.dumps(
            {
                "mode": "parity",
                "library_revision": revision,
                "cases": args.cases,
                "source_sha256": {
                    name: hashlib.sha256(source.encode()).hexdigest()
                    for name, source in sources.items()
                },
            },
            indent=2,
        )
        script = (
            f"sources={sources!r}\ncases={args.cases}\n"
            f"library_revision={revision!r}\nsubmission={submission!r}\n" + PARITY_BOOTSTRAP
        )
        (args.output / "submission.json").write_text(submission)
    compile(script, "<generated-kaggle-bootstrap>", "exec")
    (args.output / "run.py").write_text(script)
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
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
