"""Verify the downloaded frozen-v1 scope run before interpreting its answers."""

import argparse
import hashlib
import json
from pathlib import Path

from .scope_analysis import CONDITIONS


def read(path):
    return json.loads(path.read_text())


def check_results(root, protocol, banks, cold, primed, tokenizer=None):
    """Check complete paired outputs against independent pre-inference fixtures.

    These checks establish identity and completeness, not factual answer quality.
    The optional tokenizer independently decodes every stored raw sequence.
    """
    identity, completion = read(root / "identity.json"), read(root / "completion.json")
    assert identity["protocol"] == protocol
    assert identity["fingerprint"] == protocol["base_fingerprint"]
    assert identity["torch"].split("+")[0] == protocol["runtime"]["torch"]
    assert completion["status"] == "complete" and completion["new_fits"] == 0
    assert completion["cold_checks_before_new_probes"] and completion["weights_restored"]
    assert completion["final_fingerprint"] == identity["fingerprint"]
    pairs = [(r["case"], r["condition"]) for r in completion["conditions"]]
    expected_pairs = {(b["edited_case"], name) for b in banks for name in CONDITIONS}
    assert len(pairs) == len(expected_pairs) == 10 and set(pairs) == expected_pairs
    expected_cold = [
        {"case": case["case"], "condition": row["condition"], "checked": row["expected"]}
        for case in cold
        for row in case["conditions"]
    ]
    assert read(root / "cold-checks.json") == expected_cold
    outputs = []
    for bank in banks:
        case = bank["edited_case"]
        base = read(root / f"case-{case}-base.json")
        assert base["bank"] == bank and len(base["outputs"]) == len(bank["probes"]) == 8
        outputs += base["outputs"]
        for name in CONDITIONS:
            row = read(root / f"case-{case}-{name}.json")
            assert row["case"] == case and row["condition"] == name
            assert row["bank"] == bank and len(row["after"]) == len(bank["probes"])
            outputs += row["after"]
        expected_primed = [p for p in primed if p["case"] == case]
        for name in ("base", "consensus"):
            row = read(root / f"case-{case}-primed-{name}.json")
            assert row["answer_provided_in_input"] is True
            assert row["probes"] == expected_primed
            assert len(row["tails"]) == len(expected_primed) == 3
            outputs += row["tails"]
    assert len(outputs) == 108
    for row in outputs:
        assert isinstance(row["text"], str) and type(row["truncated"]) is bool
        assert 0 < len(row["tokens"]) <= 40
        assert all(type(token) is int and token >= 0 for token in row["tokens"])
        if tokenizer is not None:
            assert tokenizer.decode(row["tokens"], skip_special_tokens=True) == row["text"]
            assert row["truncated"] == (
                len(row["tokens"]) == 40 and row["tokens"][-1] != tokenizer.eos_token_id
            )
    return {"output_sequences_checked": len(outputs), "cold_sequences_checked": 20}


def verify(directory, source, tokenizer=None):
    here = Path(__file__).parent
    protocol = read(here / "scope_execution_v1.json")
    expected_submission = read(here / "scope_submission_v1.json")
    submission = read(directory / "submission.json")
    assert submission["file_sha256"] == expected_submission["file_sha256"]
    launch = read(here / "scope_launch_v1.json")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == launch["submitted_source_sha256"]
    assert len(source.read_bytes()) == launch["submitted_source_bytes"]
    for local, uploaded in (
        ("scope_inputs_v1.json", "inputs.json"),
        ("scope_protocol_v1.json", "scope_protocol.json"),
        ("scope_cold_v1.json", "cold.json"),
        ("scope_primed_v1.json", "primed.json"),
    ):
        assert (
            hashlib.sha256((here / local).read_bytes()).hexdigest()
            == protocol["data_sha256"][uploaded]
        )
    assert read(directory / "provider.json")["returncode"] == 0
    counts = check_results(
        directory / "results",
        protocol,
        read(here / "scope_inputs_v1.json"),
        read(here / "scope_cold_v1.json"),
        read(here / "scope_primed_v1.json"),
        tokenizer,
    )
    log = (directory / "execution.log").read_text()
    assert log.count("COLD_VERIFIED ") == 10
    assert log.index("COLD_VERIFIED ") < log.index("STAGE cold_load_complete_before_new_probes")
    assert log.rindex("COLD_VERIFIED ") < log.index("STAGE cold_load_complete_before_new_probes")
    assert log.index("STAGE cold_load_complete_before_new_probes") < log.index("SCOPE ")
    assert "COMPLETE frozen-edit subject scope diagnostic" in log
    return {
        "status": "Frozen-v1 scope output integrity verified; no quality claim",
        **counts,
        "token_decoding_checked": tokenizer is not None,
        "new_fits": 0,
        "executed_source_sha256": source_hash,
        "submission_metadata_matches_original": submission == expected_submission,
        "file_sha256": {
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        },
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--executed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true", help="Use the cached pinned tokenizer")
    args = parser.parse_args()
    tokenizer = None
    if args.decode:
        from transformers import GPT2Tokenizer

        protocol = read(Path(__file__).parent / "scope_execution_v1.json")
        tokenizer = GPT2Tokenizer.from_pretrained(
            protocol["model"], revision=protocol["revision"], local_files_only=True
        )
    result = verify(args.download, args.executed_source, tokenizer)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
