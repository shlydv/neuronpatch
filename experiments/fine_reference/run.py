"""Run the pinned upstream FiNE editor; no NeuronPatch training code is used.

The import shim bypasses unrelated EasyEdit algorithms, not FiNE implementations.
CUDA is required for real editing. --preflight checks upstream model parity on CPU.
Metrics here are explicit diagnostics, not the complete paper evaluation suite.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

COMMIT = "a2489cbac0dfe5a956385734065cfa041384e7d1"
MODEL = "EleutherAI/gpt-j-6B"
MODEL_REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"
DATA_REVISION = "6d218881e60c7c8be836bd2befd598eb3b318be4"
DATA_HASH = "efc01ba398dfe04975939f729aab45523addb3c6b3cf41d7429ba5de1d45225b"


def load_upstream(root):
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMIT:
        raise ValueError(f"Expected upstream {COMMIT}, found {commit}")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"], text=True):
        raise ValueError("Upstream tracked sources must be unchanged")
    package = root / "knowledge-editing" / "easyeditor"
    for suffix in ["", ".models", ".models.fine", ".util", ".modified_models"]:
        name = "easyeditor" + suffix
        module = types.ModuleType(name)
        module.__path__ = [
            str(package.joinpath(*suffix.strip(".").split("."))) if suffix else str(package)
        ]
        sys.modules[name] = module
    fine = importlib.import_module("easyeditor.models.fine.fine_main")
    model_module = importlib.import_module("easyeditor.modified_models.modeling_gptj")
    return fine, model_module._GPTJForCausalLM


def preflight(root):
    import torch
    import transformers
    from transformers import GPTJConfig, GPTJForCausalLM

    fine, model_class = load_upstream(root)
    torch.manual_seed(42)
    config = GPTJConfig(
        n_layer=2, n_embd=32, n_head=4, n_inner=64, rotary_dim=8, n_positions=32, vocab_size=101
    )
    model = model_class(config).eval()
    reference = GPTJForCausalLM(config).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[1, 8, 3, 9]])
    with torch.no_grad():
        model.set_get_activations()
        actual = model(ids).logits
        expected = reference(ids).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert len(model.activations) == 2
    assert all(tuple(a.shape) == (1, 4, 64) for a in model.activations)
    hp = fine.FINEHyperParams.from_hparams(
        str(root / "knowledge-editing/hparams/FINE/gptj-6b.yaml")
    )
    assert hp.neuron_num == 5 and hp.epochs == 50 and hp.last_layer == 3
    return {
        "status": "passed",
        "scope": (
            "imports, published configuration, CPU model/logit/activation parity; "
            "no editing or effectiveness test"
        ),
        "upstream_commit": COMMIT,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "max_logit_difference": float((actual - expected).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cases", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.preflight:
        result = preflight(args.upstream)
        (args.output / "preflight.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    import torch
    import transformers
    from transformers import GPT2Tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Published editor requires CUDA; refusing a silent device/precision substitution"
        )
    if not 1 <= args.cases <= 839:
        raise ValueError("cases must be 1..839")
    fine, model_class = load_upstream(args.upstream)
    url = f"https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/{DATA_REVISION}/benchmark/wiki_counterfact/test_cf.json"
    raw = urllib.request.urlopen(url).read()
    assert hashlib.sha256(raw).hexdigest() == DATA_HASH
    records = json.loads(raw)[: args.cases]
    hp = fine.FINEHyperParams.from_hparams(
        str(args.upstream / "knowledge-editing/hparams/FINE/gptj-6b.yaml")
    )
    hp.model_name = MODEL
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model_class.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    manifest = {
        "upstream_commit": COMMIT,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
        "data_sha256": DATA_HASH,
        "cases": args.cases,
        "case_selection": "first N in official file, no outcome filtering",
        "hparams": vars(hp),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scope": (
            "bounded upstream-editor comparison; supplementary generation diagnostics, "
            "not full paper metrics"
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def generate(prompt):
        ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        ids = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), ids], dim=-1).cuda()
        with torch.no_grad():
            output = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=40,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, ids.shape[-1] :], skip_special_tokens=True)

    for i, record in enumerate(records):
        probes = [{"kind": "rewrite", "prompt": record["prompt"], "gold": record["target_new"]}]
        if record.get("rephrase"):
            probes.append(
                {"kind": "rephrase", "prompt": record["rephrase"], "gold": record["target_new"]}
            )
        for family in ("locality", "portability"):
            for category, rows in record.get(family, {}).items():
                probes.extend(
                    {
                        "kind": f"{family}/{category}",
                        "prompt": row["prompt"],
                        "gold": row["ground_truth"],
                    }
                    for row in rows
                )
        baseline = [generate(p["prompt"]) for p in probes]
        for random_selection in (False, True):
            hp.random = random_selection
            request = {key: record[key] for key in ("prompt", "subject", "target_new")}
            model, originals = fine.apply_fine_to_model(
                model, tokenizer, [request], hp, return_orig_weights=True, keep_original_weight=True
            )
            try:
                edited = [generate(p["prompt"]) for p in probes]
                result = {
                    "case": i,
                    "random": random_selection,
                    "request": request,
                    "probes": probes,
                    "baseline": baseline,
                    "edited": edited,
                }
                (args.output / f"case-{i:03d}-random-{random_selection}.json").write_text(
                    json.dumps(result, indent=2)
                )
            finally:
                with torch.no_grad():
                    for name, value in originals.items():
                        model.get_parameter(name).copy_(value)
                del originals
            assert generate(record["prompt"]) == baseline[0], (
                "Base output changed after restoring edit"
            )


if __name__ == "__main__":
    main()
