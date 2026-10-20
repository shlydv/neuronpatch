"""Bundle one private Kaggle pilot; never include model weights or credentials."""

import argparse
import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = r"""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/tmp/fine-transfer-v1');root.mkdir(exist_ok=True)
out=Path('/kaggle/working/transfer-results');out.mkdir(exist_ok=False)
for name,value in bundle['files'].items():
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b85decode(value)
 assert hashlib.sha256(raw).hexdigest()==bundle['hashes'][name]
 path.write_bytes(raw)
(out/'submission.json').write_text(json.dumps(bundle['submission'],indent=2)+'\n')
runtime=json.loads((root/'data/protocol.json').read_text())['runtime']
commands=[
 [sys.executable,'-m','pip','install','--quiet','uv'],
 ['uv','venv','--python','3.11','/tmp/fine-transfer-env'],
 ['uv','pip','install','--python','/tmp/fine-transfer-env/bin/python',
  '--index-url','https://pypi.org/simple',*[k+'=='+v for k,v in runtime.items()]],
 ['timeout','--signal=TERM','5700','/tmp/fine-transfer-env/bin/python','-u',
  '-m','experiments.fine_transfer.run','--data',str(root/'data'),'--output',str(out/'results')],
]
env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+str(root/'src')
env['HF_HOME']='/tmp/fine-transfer-hf';env['HF_HUB_DISABLE_XET']='1'
started=time.monotonic();returncode=1
provider={'provider':'Kaggle','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)}
try:
 with (out/'execution.log').open('w') as log:
  for command in commands:
   print('COMMAND',command[:6],flush=True)
   process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,bufsize=1,env=env)
   for line in process.stdout:
    log.write(line);log.flush();print(line,end='',flush=True)
   returncode=process.wait()
   if returncode:raise RuntimeError(f'Worker command returned {returncode}')
finally:
 provider.update(returncode=returncode,elapsed_seconds=time.monotonic()-started)
 (out/'provider.json').write_text(json.dumps(provider,indent=2)+'\n')
 print('PILOT_EXIT',returncode,flush=True)
"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "src/neuronpatch").glob("*.py"))
    sources += [
        ROOT / "experiments/fine_transfer" / name for name in ("run.py", "fit.py", "gradient.py")
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
        "scope": "One bounded eight-fit development pilot",
    }
    bundle = {
        "files": {k: base64.b85encode(v).decode() for k, v in files.items()},
        "hashes": hashes,
        "submission": submission,
    }
    packed = base64.b85encode(zlib.compress(json.dumps(bundle).encode(), 9)).decode()
    script = (
        "import base64,hashlib,json,zlib\n"
        f"bundle=json.loads(zlib.decompress(base64.b85decode({packed!r})))\n" + BOOTSTRAP
    )
    compile(script, "<fine-transfer-kaggle>", "exec")
    (args.output / "run.py").write_text(script)
    metadata = {
        "id": "shlydv/neuronpatch-fine-gradient-transfer-v1-20261020",
        "title": "neuronpatch fine gradient transfer v1 20261020",
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
    (args.output / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
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
