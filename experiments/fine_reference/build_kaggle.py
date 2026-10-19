"""Package the reference harness for Kaggle's current Python3.13 host image."""

import argparse
import json
from pathlib import Path

BOOTSTRAP = """import subprocess, sys
from pathlib import Path
root=Path('/kaggle/working/fine-reference-harness');root.mkdir(exist_ok=True)
for name,source in sources.items(): (root/name).write_text(source)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv'],check=True)
subprocess.run(['uv','venv','--python','3.11','/kaggle/working/fine-env'],check=True)
python='/kaggle/working/fine-env/bin/python'
subprocess.run(['uv','pip','install','--python',python,'--index-url',
    'https://pypi.org/simple','torch==2.0.1','numpy==1.26.4',
    'transformers==4.40.0','tokenizers==0.19.1','huggingface-hub==0.23.5',
    'accelerate==0.30.1','PyYAML','requests','tqdm'],check=True)
repo='/kaggle/working/fine-upstream'
subprocess.run(['git','clone','--quiet',
    'https://github.com/opanhw/fine-unified-editing-code',repo],check=True)
subprocess.run(['git','-C',repo,'checkout','--quiet',
    'a2489cbac0dfe5a956385734065cfa041384e7d1'],check=True)
subprocess.run([python,'-u',str(root/'run.py'),'--upstream',repo,
    '--output','/kaggle/working/preflight','--preflight'],check=True)
subprocess.run([python,'-u',str(root/'run.py'),'--upstream',repo,
    '--output','/kaggle/working/results','--cases',str(cases)],check=True)
"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--slug", default="neuronpatch-fine-reference-smoke-20261019")
    args = parser.parse_args()
    if not 1 <= args.cases <= 839:
        parser.error("cases must be 1..839")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    sources = {name: (root / name).read_text() for name in ("run.py", "upstream_metrics.py")}
    script = f"sources={sources!r}\ncases={args.cases}\n" + BOOTSTRAP
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
