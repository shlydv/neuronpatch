"""Bounded fixed-support objective ablation; no selector or support tuning."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import GPT2Tokenizer, GPTJForCausalLM

from experiments.fine_transfer.run import generate, merge, restore, save_columns, write
from experiments.fine_transfer.scope_analysis import digest_output
from neuronpatch.editing import GPTJEditConfig
from neuronpatch.occurrence import OccurrenceColumnEdit
from neuronpatch.patch import fingerprint

from .analysis import select_weight
from .fit import capture_calibration, fit_binding
from .measure import measure_merged


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.data / "protocol.json").read_text())
    for name, expected in protocol["data_sha256"].items():
        assert hashlib.sha256((args.data / name).read_bytes()).hexdigest() == expected, name
    inputs = json.loads((args.data / "inputs.json").read_text())
    cold = json.loads((args.data / "cold.json").read_text())
    write(args.output / "inputs.json", inputs)
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > protocol["runtime_limit_seconds"]:
            raise TimeoutError("Bounded subject-binding worker exceeded runtime limit")

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
            "protocol": protocol,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    )
    targets = {}
    verified = []
    for c in cold:
        case = c["case"]
        check_time()
        before = [digest_output(generate(model, tokenizer, p["prompt"])) for p in c["probes"]]
        assert before == [p["base"] for p in c["probes"]]
        edit = OccurrenceColumnEdit.load(model, args.data / f"case-{case}-reference")
        targets[case] = list(edit.targets)
        delta = edit.delta.detach().cpu().clone()
        del edit
        saved = save_columns(model, targets[case])
        try:
            merge(model, targets[case], delta)
            actual = [digest_output(generate(model, tokenizer, p["prompt"])) for p in c["probes"]]
            assert actual == [p["edited"] for p in c["probes"]]
        finally:
            restore(model, saved)
        assert digest_output(generate(model, tokenizer, c["probes"][0]["prompt"])) == before[0]
        verified.append({"case": case, "base": before, "edited": actual})
        write(args.output / "cold-checks.json", verified)
        print("COLD_VERIFIED", case, flush=True)
    assert fingerprint(model) == identity
    print("STAGE cold_complete_before_new_probes", flush=True)
    fit_config = protocol["fit"]
    config = GPTJEditConfig(
        max_steps=fit_config["steps"],
        lr=fit_config["lr"],
        alpha=fit_config["alpha"],
        beta=fit_config["beta"],
        gamma=fit_config["gamma"],
        early_stop_probability=None,
    )
    fits = []
    for case in inputs["cases"]:
        c = case["case"]
        check_time()

        def evaluate():
            return {
                "case": c,
                **{
                    name: [generate(model, tokenizer, p["prompt"]) for p in probes]
                    for name, probes in [
                        ("edit", case["edit_probes"]),
                        ("utility", inputs["utility"]),
                        *(
                            (name, case[name])
                            for name in ("calibration", "evaluation", "exposed_scope")
                        ),
                    ]
                },
            }

        base = evaluate()
        write(args.output / f"case-{c}-base.json", base)
        calibration = [p["prompt"] for p in case["calibration"]]
        protected = capture_calibration(model, tokenizer, calibration)
        reference = {
            "logits": torch.cat([p[2].cpu() for p in protected]),
            "lengths": torch.tensor([len(p[2]) for p in protected]),
        }
        save_file(reference, args.output / f"case-{c}-calibration-base.safetensors")
        for weight in protocol["preservation_weights"]:
            check_time()
            print("FIT_START", c, weight, flush=True)
            fitted = time.monotonic()
            edit, history, live_final = fit_binding(
                model,
                tokenizer,
                case["train_prompts"],
                case["target"],
                targets[c],
                calibration,
                weight,
                config,
            )
            assert len(history) == fit_config["steps"]
            assert list(edit.targets) == targets[c]
            assert not any(p.requires_grad for p in model.parameters())
            delta = edit.delta.detach().cpu().clone()
            artifact = args.output / f"case-{c}-w{int(weight)}-edit"
            edit.save(
                artifact,
                metadata={
                    "case": c,
                    "preservation_weight": weight,
                    "fixed_updates": config.max_steps,
                },
            )
            scalars = edit.num_trainable_parameters
            del edit
            saved = save_columns(model, targets[c])
            try:
                merge(model, targets[c], delta)
                final, calibration_tensors = measure_merged(
                    model, tokenizer, case["train_prompts"], case["target"], protected, reference
                )
                save_file(
                    calibration_tensors,
                    args.output / f"case-{c}-w{int(weight)}-calibration.safetensors",
                )
                outputs = evaluate()
                write(args.output / f"case-{c}-w{int(weight)}.json", outputs)
            finally:
                restore(model, saved)
            assert generate(model, tokenizer, case["edit_probes"][0]["prompt"]) == base["edit"][0]
            fits.append(
                {
                    "case": c,
                    "weight": weight,
                    "updates": config.max_steps,
                    "final": final,
                    "live_final": live_final,
                    "history": history,
                    "trainable_scalars": scalars,
                    "seconds": time.monotonic() - fitted,
                }
            )
            write(args.output / "fits.json", fits)
            print("FIT_RESTORED", c, weight, "final", final, flush=True)
            del delta, outputs
            gc.collect()
            torch.cuda.empty_cache()
    assert len(fits) == protocol["new_fits"] == 6
    selected = select_weight(
        fits,
        protocol["preservation_weights"],
        [c["case"] for c in inputs["cases"]],
        protocol["selection"],
    )
    assert fingerprint(model) == identity
    write(
        args.output / "completion.json",
        {
            "status": "complete",
            "new_fits": len(fits),
            "selected_weight": selected,
            "selection_uses_calibration_only": True,
            "weights_restored": True,
            "final_fingerprint": identity,
            "seconds": time.monotonic() - started,
        },
    )
    print("COMPLETE subject-binding objective ablation", flush=True)


if __name__ == "__main__":
    main()
