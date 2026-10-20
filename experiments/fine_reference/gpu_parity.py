"""Compare the library and unchanged FiNE on the same full-size checkpoint.

Two sequential model loads avoid holding two 6B models on the GPU. This is a
port-validation experiment, not evidence that either method is better. The
isolated runtime must support the pinned upstream Transformers implementation.
"""

import argparse
import gc
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from run import COMMIT, DATA_HASH, DATA_REVISION, MODEL, MODEL_REVISION, load_upstream
from transformers import GPT2Tokenizer, GPTJForCausalLM

from neuronpatch.editing import GPTJEditConfig, fit_gptj_edit
from neuronpatch.occurrence import OccurrenceColumnEdit


@torch.no_grad()
def generate(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    ids = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), ids], dim=-1).cuda()
    output = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, ids.shape[-1] :], skip_special_tokens=True)


def probes(record):
    values = [{"kind": "rewrite", "prompt": record["prompt"]}]
    if record.get("rephrase"):
        values.append({"kind": "rephrase", "prompt": record["rephrase"]})
    # Fixed first alias and first probe in each locality category. No output filtering.
    for family in ("portability", "locality"):
        for kind, rows in record.get(family, {}).items():
            if rows and (family == "locality" or kind == "Subject_Aliasing"):
                values.append({"kind": f"{family}/{kind}", "prompt": rows[0]["prompt"]})
    return values


def columns(model, targets):
    by_name = defaultdict(set)
    for name, feature in targets:
        by_name[name].add(feature)
    return {
        name: (
            sorted(features),
            model.get_submodule(name).weight[:, sorted(features)].detach().cpu(),
        )
        for name, features in by_name.items()
    }


@torch.no_grad()
def add_updates(model, targets, delta):
    for row, (name, feature) in enumerate(targets):
        weight = model.get_submodule(name).weight
        weight[:, feature].add_(delta[row].to(weight.device))


@torch.no_grad()
def restore_columns(model, saved):
    for name, (indices, values) in saved.items():
        weight = model.get_submodule(name).weight
        weight[:, indices] = values.to(weight.device)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full-size parity check requires CUDA")
    if args.case_start < 0 or not 1 <= args.cases <= 2 or args.case_start + args.cases > 839:
        raise ValueError("Use one or two consecutive official cases")
    args.output.mkdir(parents=True, exist_ok=False)
    fine, upstream_class = load_upstream(args.upstream)
    url = (
        f"https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/{DATA_REVISION}"
        "/benchmark/wiki_counterfact/test_cf.json"
    )
    raw = urllib.request.urlopen(url).read()
    if hashlib.sha256(raw).hexdigest() != DATA_HASH:
        raise ValueError("Dataset hash mismatch")
    all_records = json.loads(raw)
    cases = list(range(args.case_start, args.case_start + args.cases))
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    hp = fine.FINEHyperParams.from_hparams(
        str(args.upstream / "knowledge-editing/hparams/FINE/gptj-6b.yaml")
    )
    hp.model_name = MODEL
    manifest = {
        "upstream_commit": COMMIT,
        "library_revision": args.library_revision,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
        "data_sha256": DATA_HASH,
        "cases": cases,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "hparams": vars(hp).copy(),
        "criteria": {
            "same_base_generations": "exact",
            "same_supports": "exact ordered occurrences",
            "same_learned_deltas": "exact, otherwise report the discrepancy",
            "same_merged_columns": "exact",
            "same_merged_generations": "exact on fixed probes",
            "round_trip": "exact deltas and live generations",
        },
        "scope": "full-size port parity; no improvement or generalization claim",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries = []
    for implementation, model_class in (("upstream", upstream_class), ("library", GPTJForCausalLM)):
        print(f"STAGE parity load={implementation}", flush=True)
        model = model_class.from_pretrained(
            MODEL, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map={"": 0}
        ).eval()
        for case in cases:
            record = all_records[case]
            fixed_probes = probes(record)
            before = [generate(model, tokenizer, p["prompt"]) for p in fixed_probes]
            for random_selection in (False, True):
                label = f"case-{case:03d}-random-{random_selection}"
                print(f"STAGE parity {implementation} {label}", flush=True)
                request = {k: record[k] for k in ("prompt", "subject", "target_new")}
                if implementation == "upstream":
                    hp.random = random_selection
                    torch.manual_seed(42)
                    delta, selected, original_layers = fine.execute_fine(
                        model, tokenizer, request, hp, 0
                    )
                    targets = [
                        (f"transformer.h.{layer}.mlp.fc_out", feature)
                        for layer, feature in selected.long().tolist()
                    ]
                    # execute_fine returns full copies; scoped column copies suffice here.
                    del original_layers
                    artifact = None
                    history = None
                else:
                    result = fit_gptj_edit(
                        model,
                        tokenizer,
                        request["prompt"],
                        request["target_new"],
                        GPTJEditConfig(selection="random" if random_selection else "contribution"),
                    )
                    delta = result.edit.delta.detach()
                    targets = list(result.edit.targets)
                    history = result.history
                    artifact = args.output / f"{label}-edit"
                    result.edit.save(artifact, metadata=manifest)
                    with result.edit.applied():
                        live = [generate(model, tokenizer, p["prompt"]) for p in fixed_probes]
                    expected_delta = delta.cpu().clone()
                    # Reload against this same untouched base; no second full model is needed.
                    del result
                    loaded = OccurrenceColumnEdit.load(model, artifact)
                    torch.testing.assert_close(loaded.delta.cpu(), expected_delta, atol=0, rtol=0)
                    with loaded.applied():
                        reloaded = [generate(model, tokenizer, p["prompt"]) for p in fixed_probes]
                    if live != reloaded:
                        raise AssertionError("Compact reload changed live generations")
                    del loaded
                saved = columns(model, targets)
                merged, state = None, None
                try:
                    add_updates(model, targets, delta)
                    after = [generate(model, tokenizer, p["prompt"]) for p in fixed_probes]
                    merged = columns(model, targets)
                    state = {
                        "targets": targets,
                        "delta": delta.detach().cpu(),
                        "merged": merged,
                    }
                    if implementation == "upstream":
                        torch.save(state, args.output / f"{label}-upstream.pt")
                        record_out = {
                            "case": case,
                            "random": random_selection,
                            "request": request,
                            "probes": fixed_probes,
                            "before": before,
                            "after": after,
                        }
                        (args.output / f"{label}-upstream.json").write_text(
                            json.dumps(record_out, indent=2)
                        )
                    else:
                        expected = torch.load(
                            args.output / f"{label}-upstream.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        previous = json.loads((args.output / f"{label}-upstream.json").read_text())
                        same_targets = targets == expected["targets"]
                        same_delta = same_targets and torch.equal(state["delta"], expected["delta"])
                        same_merged = same_targets and all(
                            merged[name][0] == indices and torch.equal(merged[name][1], values)
                            for name, (indices, values) in expected["merged"].items()
                        )
                        summary = {
                            "case": case,
                            "random": random_selection,
                            "same_base_generations": before == previous["before"],
                            "same_supports": same_targets,
                            "same_learned_deltas": same_delta,
                            "max_delta_difference": (
                                float((state["delta"] - expected["delta"]).abs().max())
                                if same_targets
                                else None
                            ),
                            "same_merged_columns": same_merged,
                            "same_merged_generations": after == previous["after"],
                            "compact_round_trip": True,
                            "probes": fixed_probes,
                            "before": before,
                            "upstream_after": previous["after"],
                            "library_after": after,
                            "library_live": live,
                            "history": history,
                        }
                        summaries.append(summary)
                        (args.output / f"{label}-comparison.json").write_text(
                            json.dumps(summary, indent=2)
                        )
                finally:
                    restore_columns(model, saved)
                    del delta, saved, merged, state
                if [generate(model, tokenizer, p["prompt"]) for p in fixed_probes] != before:
                    raise AssertionError("Base generations changed after restoring columns")
                print(f"STAGE parity restored {implementation} {label}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    keys = (
        "same_base_generations",
        "same_supports",
        "same_learned_deltas",
        "same_merged_columns",
        "same_merged_generations",
        "compact_round_trip",
    )
    passed = all(all(row[key] for key in keys) for row in summaries)
    (args.output / "summary.json").write_text(
        json.dumps({"passed": passed, "comparisons": summaries}, indent=2)
    )
    print(f"STAGE parity complete exact={passed}", flush=True)
    if not passed:
        raise AssertionError("Parity discrepancy recorded; inspect comparisons before adapting")


if __name__ == "__main__":
    main()
