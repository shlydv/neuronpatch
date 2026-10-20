"""Bundle exact experiment sources/data and a streaming Colab T4 launcher."""

import argparse
import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BOOTSTRAP = r"""
import datetime, hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path
root=Path('/tmp/neuronpatch-evidence-v1');root.mkdir(exist_ok=True)
out=Path('/content/evidence-pairs-v1');out.mkdir(exist_ok=False)
for name,value in sources.items():
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
for name,value in data.items():
    path=root/'frozen-data'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
(out/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
provider={'provider':'Google Colab standard T4',
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'host_python':sys.version,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()}
(out/'provider.json').write_text(json.dumps(provider,indent=2)+'\n')
print(json.dumps(provider,indent=2),flush=True)
commands=[
    [sys.executable,'-m','pip','install','--quiet','uv'],
    ['uv','venv','--python','3.11','/tmp/evidence-env'],
    ['uv','pip','install','--python','/tmp/evidence-env/bin/python','--index-url','https://pypi.org/simple',
     *[name+'=='+version for name,version in runtime.items()]],
    ['/tmp/evidence-env/bin/python','-u','-m','experiments.evidence_pairs.preflight','--output',str(out/'preflight')],
    ['/tmp/evidence-env/bin/python','-u','-m','experiments.evidence_pairs.run',
     '--data',str(root/'frozen-data'),'--protocol',str(root/'experiments/evidence_pairs/protocol_v1.json'),
     '--output',str(out/'results')],
]
env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+str(root/'src');env['HF_HUB_DISABLE_XET']='1'
started=time.monotonic();returncode=1
try:
    with (out/'execution.log').open('w') as log:
        for command in commands:
            print('COMMAND',command[:6],flush=True)
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            returncode=process.wait()
            if returncode: raise RuntimeError(f'Experiment command exited {returncode}')
    with (out/'runtime_freeze.txt').open('w') as stream:
        subprocess.run(['uv','pip','freeze','--python','/tmp/evidence-env/bin/python'],stdout=stream,check=True)
finally:
    provider.update(elapsed_seconds=time.monotonic()-started,returncode=returncode)
    (out/'provider.json').write_text(json.dumps(provider,indent=2)+'\n')
    archive=Path('/content/evidence_pairs_v1_outputs.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in out.rglob('*'):
            if path.is_file(): zipped.write(path,path.relative_to(out))
    print('OUTPUT_ARCHIVE',str(archive),archive.stat().st_size,'RETURN_CODE',returncode,flush=True)
"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT / "src/neuronpatch").glob("*.py"))
    paths += [
        ROOT / "experiments/evidence_pairs" / name
        for name in ("run.py", "preflight.py", "objective.py", "protocol_v1.json")
    ]
    paths += [ROOT / "experiments/protected_rows/helpers.py"]
    paths += [ROOT / "experiments/real_qa" / name for name in ("data.py", "response_loss.py")]
    sources = {str(p.relative_to(ROOT)): p.read_text() for p in paths}
    data = {p.name: p.read_text() for p in args.data.glob("*.json")}
    protocol = json.loads((ROOT / "experiments/evidence_pairs/protocol_v1.json").read_text())

    def sha(value):
        return hashlib.sha256(value.encode()).hexdigest()

    assert {name: sha(value) for name, value in data.items()} == protocol["data_sha256"]
    submission = {
        "revision_at_packaging": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {name: sha(value) for name, value in sources.items()},
        "data_sha256": protocol["data_sha256"],
        "scope": protocol["scope"],
    }
    script = (
        f"sources={sources!r}\ndata={data!r}\nruntime={protocol['runtime']!r}\n"
        f"submission={submission!r}\n" + BOOTSTRAP
    )
    compile(script, "<evidence-pairs-v1>", "exec")
    (args.output / "run_colab.py").write_text(script)
    packed = base64.b85encode(zlib.compress(script.encode(), level=9)).decode()
    cell = (
        "# Frozen NeuronPatch evidence-pair experiment, source SHA256 " + sha(script) + "\n"
        "import base64, hashlib, zlib\n"
        f"experiment = zlib.decompress(base64.b85decode({packed!r})).decode()\n"
        f"assert hashlib.sha256(experiment.encode()).hexdigest() == {sha(script)!r}\n"
        "exec(compile(experiment, '<neuronpatch-evidence-v1>', 'exec'))\n"
    )
    (args.output / "cell.py").write_text(cell)
    (args.output / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    print(json.dumps({"bytes": len(script), "sha256": sha(script)}))


if __name__ == "__main__":
    main()
