"""Cold-load frozen edits and test whether their answers leak to other subjects."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPTJForCausalLM

from neuronpatch.occurrence import OccurrenceColumnEdit
from neuronpatch.patch import fingerprint

from .run import generate, merge, restore, save_columns, write
from .scope_analysis import digest_output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.data / "execution.json").read_text())
    for name, expected in protocol["data_sha256"].items():
        assert hashlib.sha256((args.data / name).read_bytes()).hexdigest() == expected, name
    banks = json.loads((args.data / "inputs.json").read_text())
    cold = json.loads((args.data / "cold.json").read_text())
    primed = json.loads((args.data / "primed.json").read_text())
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Frozen no-training scope runtime limit exceeded")

    print("STAGE model_load", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = (
        GPTJForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        .eval()
        .requires_grad_(False)
    )
    identity = fingerprint(model)
    assert identity == protocol["base_fingerprint"]
    write(
        args.output / "identity.json",
        {
            "fingerprint": identity,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "protocol": protocol,
        },
    )
    print("STAGE cold_load_checks", flush=True)
    cached, cold_decisions = {}, []
    for case in cold:
        case_id = case["case"]
        for probe in case["probes"]:
            assert digest_output(generate(model, tokenizer, probe["prompt"])) == probe["base"]
        for record in case["conditions"]:
            check_time()
            name = record["condition"]
            edit = OccurrenceColumnEdit.load(model, args.data / record["artifact"])
            assert not any(p.requires_grad for p in model.parameters())
            targets = list(edit.targets)
            delta = edit.delta.detach().cpu().clone()
            assert delta.dtype == torch.bfloat16
            assert edit.num_trainable_parameters == record["trainable_scalars"]
            del edit
            saved = save_columns(model, targets)
            try:
                merge(model, targets, delta)
                actual = [
                    digest_output(generate(model, tokenizer, p["prompt"])) for p in case["probes"]
                ]
                assert actual == record["expected"], (case_id, name, "Cold-load mismatch")
            finally:
                restore(model, saved)
            assert (
                digest_output(generate(model, tokenizer, case["probes"][0]["prompt"]))
                == case["probes"][0]["base"]
            )
            cached[case_id, name] = (targets, delta)
            cold_decisions.append({"case": case_id, "condition": name, "checked": actual})
            write(args.output / "cold-checks.json", cold_decisions)
            gc.collect()
            torch.cuda.empty_cache()
            print("COLD_VERIFIED", case_id, name, flush=True)
    assert fingerprint(model) == identity
    print("STAGE cold_load_complete_before_new_probes", flush=True)
    completed = []
    for bank in banks:
        case_id = bank["edited_case"]
        check_time()
        before = [generate(model, tokenizer, p["prompt"]) for p in bank["probes"]]
        write(args.output / f"case-{case_id}-base.json", {"bank": bank, "outputs": before})
        forced = [p for p in primed if p["case"] == case_id]
        # These answer prefixes are provided in the input. They never count as edit successes.
        forced_base = [generate(model, tokenizer, p["forced_prompt"]) for p in forced]
        write(
            args.output / f"case-{case_id}-primed-base.json",
            {"probes": forced, "tails": forced_base, "answer_provided_in_input": True},
        )
        for record in next(c for c in cold if c["case"] == case_id)["conditions"]:
            check_time()
            name = record["condition"]
            targets, delta = cached[case_id, name]
            saved = save_columns(model, targets)
            print("SCOPE", case_id, name, flush=True)
            try:
                merge(model, targets, delta)
                after = [generate(model, tokenizer, p["prompt"]) for p in bank["probes"]]
                write(
                    args.output / f"case-{case_id}-{name}.json",
                    {"case": case_id, "condition": name, "bank": bank, "after": after},
                )
                if name == "gradient_consensus":
                    write(
                        args.output / f"case-{case_id}-primed-consensus.json",
                        {
                            "probes": forced,
                            "tails": [
                                generate(model, tokenizer, p["forced_prompt"]) for p in forced
                            ],
                            "answer_provided_in_input": True,
                        },
                    )
            finally:
                restore(model, saved)
            assert generate(model, tokenizer, bank["probes"][0]["prompt"]) == before[0]
            completed.append({"case": case_id, "condition": name})
            print("RESTORED", case_id, name, flush=True)
    assert len(completed) == 10
    assert fingerprint(model) == identity
    write(
        args.output / "completion.json",
        {
            "status": "complete",
            "conditions": completed,
            "new_fits": 0,
            "cold_checks_before_new_probes": True,
            "weights_restored": True,
            "final_fingerprint": identity,
            "seconds": time.monotonic() - started,
        },
    )
    print("COMPLETE frozen-edit subject scope diagnostic", flush=True)


if __name__ == "__main__":
    main()
