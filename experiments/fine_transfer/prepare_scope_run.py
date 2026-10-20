"""Freeze saved v1 patches and a separately labeled continuation control."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .scope_analysis import CONDITIONS, digest_output


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot-download", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    protocol = json.loads((here / "protocol_v1.json").read_text())
    scope = json.loads((here / "scope_protocol_v1.json").read_text())
    integrity = json.loads((here / "pilot_v1/integrity.json").read_text())
    for name, expected in integrity["file_sha256"].items():
        assert sha(args.pilot_download / name) == expected, name
    assert sha(here / "scope_inputs_v1.json") == scope["inputs_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(here / "scope_inputs_v1.json", args.output / "inputs.json")
    shutil.copyfile(here / "scope_protocol_v1.json", args.output / "scope_protocol.json")
    cases, primed = [], []
    results = args.pilot_download / "results"
    for case in (0, 1):
        original = json.loads((results / f"case-{case}-fine-original.json").read_text())
        probes = [
            {
                "prompt": original["probes"][i]["prompt"],
                "base": digest_output(original["before"][i]),
            }
            for i in (0, 1)
        ]
        fits = []
        for condition in CONDITIONS:
            row = json.loads((results / f"case-{case}-{condition}.json").read_text())
            if condition == "fine-original":
                artifact = args.references / f"case-{case}"
                for name in ("edit.json", "edit.safetensors"):
                    assert (
                        sha(artifact / name) == protocol["artifact_sha256"][f"case-{case}/{name}"]
                    )
            else:
                artifact = results / f"case-{case}-{condition}-edit"
            destination = f"edits/case-{case}-{condition}"
            shutil.copytree(artifact, args.output / destination)
            manifest = json.loads((artifact / "edit.json").read_text())
            fits.append(
                {
                    "condition": condition,
                    "artifact": destination,
                    "trainable_scalars": len(manifest["targets"]) * 4096,
                    "expected": [digest_output(row["after"][i]) for i in (0, 1)],
                }
            )
        cases.append({"case": case, "probes": probes, "conditions": fits})
        for probe in original["probes"]:
            if probe["kind"] in ("rephrase", "fresh_wording"):
                primed.append(
                    {
                        "case": case,
                        "kind": probe["kind"],
                        "prompt": probe["prompt"],
                        "provided_answer": probe["gold"],
                        "forced_prompt": probe["prompt"] + " " + probe["gold"],
                    }
                )
    assert len(primed) == 6
    for name, value in (("cold.json", cases), ("primed.json", primed)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    execution = {
        "id": "fine-transfer-scope-execution-v1-20261020",
        "model": protocol["model"],
        "revision": protocol["revision"],
        "base_fingerprint": protocol["base_fingerprint"],
        "runtime": protocol["runtime"],
        "runtime_limit_seconds": 2400,
        "new_fits": 0,
        "main_protocol_sha256": sha(here / "scope_protocol_v1.json"),
        "main_inputs_sha256": scope["inputs_sha256"],
        "conditions": CONDITIONS,
        "execution_precondition": (
            "Satisfied by the completed, reviewed pilot: both gradient variants retain 2/2 "
            "original prefixes, exceed augmented FiNE on untrained wordings, and retain "
            "all 8 baseline-prefix-correct global utility questions under both edits. "
            "No other GPU worker is active."
        ),
        "prediction": (
            "The broad continuation changes and shared early columns suggest actual-gradient "
            "patches may produce more off-subject new-target prefixes than original FiNE. "
            "This is a directional diagnostic prediction, not a fitted probability."
        ),
        "scope_interpretation": (
            "Any new-target prefix absent in base on another subject is observed scope "
            "failure. Report each condition and case, all reference-prefix losses and exact "
            "continuation changes. No leaks in 16 prompts would not prove specificity; "
            "an empty baseline-correct set cannot prove retention. No tuning or refitting."
        ),
        "supplemental_control": (
            "Registered after pilot output review, before scope inference: provide the "
            "same target answer in each of the six untrained wordings, then generate 40 "
            "tail tokens with base and saved consensus patch. This separates what the base "
            "does conditional on that answer from additional patch-induced changes. "
            "Answer is supplied in the input; never an editing-success score. Qualitative "
            "review and exact tail agreement cannot by themselves establish factual quality."
        ),
        "data_sha256": {
            str(p.relative_to(args.output)): sha(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps({"patches": 10, "scope_probes": 16, "primed_probes": 6, "new_fits": 0}))


if __name__ == "__main__":
    main()
