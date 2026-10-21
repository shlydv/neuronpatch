"""Freeze only the remaining interventions after a verified bounded timeout."""

import argparse
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

from .run import verify_parent
from .verify import check_results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--partial-download", type=Path, required=True)
    parser.add_argument("--partial-integrity", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--executed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified = json.loads(args.partial_integrity.read_text())
    protocol = json.loads(args.protocol.read_text())
    root = args.partial_download / "results"
    provider = json.loads((args.partial_download / "provider.json").read_text())
    assert provider["returncode"] == 1
    log = (args.partial_download / "execution.log").read_text()
    assert "TimeoutError: Frozen-adapter diagnostic exceeded runtime limit" in log
    assert not verified["complete"] and verified["decoded"] == verified["sequences_checked"]
    verify_parent(root, verified["file_sha256"])
    current = check_results(root, args.parent, protocol, partial=True)
    for key in (
        "file_sha256",
        "completed_interventions",
        "remaining_interventions",
        "sequences_checked",
        "calibration_records_recomputed",
    ):
        assert current[key] == verified[key]
    assert verified["completed_interventions"] == 16
    assert verified["remaining_interventions"] == [
        {"case": 1, "condition": "only-final-three"},
        {"case": 1, "condition": "without-final-three"},
    ]
    manifest = {
        "status": "Verified prefix of frozen grid; missing suffix only, not new settings",
        "partial_kernel": "shlydv/neuronpatch-frozen-adapter-diagnosis-v1-20261021",
        "partial_kernel_version": 1,
        "partial_source_sha256": "7fb054bfb65908f9f820c04d36add34475657bcde8f442deb59b0ed99590c317",
        "partial_integrity_sha256": hashlib.sha256(args.partial_integrity.read_bytes()).hexdigest(),
        "file_sha256": verified["file_sha256"],
        "completed_interventions": verified["completed_interventions"],
        "remaining_interventions": verified["remaining_interventions"],
        "repeated_control_observations": 212,
        "new_ablation_observations": 106,
        "new_fits": 0,
    }
    submission = json.loads((args.partial_download / "submission.json").read_text())
    raw = args.executed_source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["partial_source_sha256"]
    tree = ast.parse(raw)
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "b85decode"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    )
    bundle = json.loads(zlib.decompress(base64.b85decode(ast.literal_eval(call.args[0]))))
    assert bundle["submission"] == submission
    assert bundle["hashes"] == submission["file_sha256"]
    for name, digest in bundle["hashes"].items():
        assert hashlib.sha256(base64.b85decode(bundle["files"][name])).hexdigest() == digest
    manifest["partial_submission"] = submission
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
    (args.output / "resume.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "Frozen two remaining interventions and212 repeated controls; original protocol unchanged"
    )


if __name__ == "__main__":
    main()
