"""CPU checks of pinned, inspected NeST routines; never import its training scripts."""

import argparse
import ast
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from neuronpatch.grouped import GroupedRowPatch

PINS = {
    "1_get_safety_neuron.py": "cb96fac68ea29fb37a99e12b7b5af95505fb10baf5660989af6f9a43a2ec6118",
    "2_sn_tune_cluster.py": "b53e0b4e0c28590a742953dbee4a1e1d577a08a81a3424a7b036ebce3652d128",
    "util_model.py": "143f6ad71c4fdc897b4a476ac12475193f1ee7458351e147e25ade94ab7b6062",
}


def definitions(tree, names, namespace):
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    selected = [n for n in nodes if n.name in names]
    assert {n.name for n in selected} == set(names)
    # Exact source hashes are checked first. Only reviewed tensor/parser routines
    # are compiled; no imports, top-level statements, downloads or trainer run.
    exec(
        compile(ast.Module(selected, type_ignores=[]), "<pinned-nest-routines>", "exec"), namespace
    )


def parity(namespace, dtype):
    torch.manual_seed(811)
    base = nn.Linear(5, 7, dtype=dtype).requires_grad_(False)
    upstream = deepcopy(base)
    indices, groups = [6, 1, 3], [1, 0, 1]
    parameters, clusters = namespace["enable_clustered_training_for_linear"](
        upstream, indices, groups
    )
    compact = GroupedRowPatch(deepcopy(base), indices, groups)
    delta = torch.randn_like(compact.delta) * 0.01
    upstream_delta = upstream.parametrizations.weight.original
    with torch.no_grad():
        upstream_delta.copy_(delta)
        compact.delta.copy_(delta)
    x = torch.randn(2, 3, 5, dtype=dtype, requires_grad=True)
    other_x = x.detach().clone().requires_grad_(True)
    y, other_y = upstream(x), compact(other_x)
    atol = 1e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(y, other_y, atol=atol, rtol=atol)
    coefficients = torch.randn_like(y)
    grads = torch.autograd.grad((y * coefficients).sum(), (upstream_delta, x))
    other_grads = torch.autograd.grad((other_y * coefficients).sum(), (compact.delta, other_x))
    for a, b in zip(grads, other_grads, strict=True):
        torch.testing.assert_close(a, b, atol=atol, rtol=atol)
    assert parameters == compact.delta.numel() == 10 and clusters == 2
    assert sum(p.numel() for p in upstream.parameters() if p.requires_grad) == 10
    namespace["fold_and_remove_cluster_parametrizations"](upstream)
    merged = compact.merged_linear()
    torch.testing.assert_close(upstream.weight, merged.weight, atol=0, rtol=0)
    untouched = [0, 2, 4, 5]
    torch.testing.assert_close(upstream.weight[untouched], base.weight[untouched], atol=0, rtol=0)
    return {
        "dtype": str(dtype),
        "trainable_scalars": parameters,
        "max_forward_difference": float((y - other_y).detach().abs().max()),
        "max_update_gradient_difference": float((grads[0] - other_grads[0]).abs().max()),
        "max_input_gradient_difference": float((grads[1] - other_grads[1]).abs().max()),
        "merged_weights_exact": True,
    }


def check(source):
    trees = {}
    for name, expected in PINS.items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        trees[name] = ast.parse(data.decode())
    namespace = {
        "torch": torch,
        "nn": nn,
        "parametrize": parametrize,
        "Iterable": Iterable,
        "Tuple": Tuple,
        "Dict": Dict,
    }
    definitions(
        trees["2_sn_tune_cluster.py"],
        {
            "ClusteredRowParametrize",
            "enable_clustered_training_for_linear",
            "fold_and_remove_cluster_parametrizations",
        },
        namespace,
    )
    results = {"row_update_parity": [parity(namespace, d) for d in (torch.float64, torch.float32)]}

    # A numerical-method distinction, not evidence about a pretrained model.
    base = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        base.weight.fill_(1)
    upstream = deepcopy(base)
    namespace["enable_clustered_training_for_linear"](upstream, [0], [0])
    compact = GroupedRowPatch(deepcopy(base), [0], [0])
    with torch.no_grad():
        delta = torch.tensor([[0.0001, 0]], dtype=torch.bfloat16)
        upstream.parametrizations.weight.original.copy_(delta)
        compact.delta.copy_(delta)
    x = torch.tensor([[1, -1]], dtype=torch.bfloat16)
    a, b = float(upstream(x).detach()), float(compact(x).detach())
    assert a == 0 and b > 0 and float(compact.merged_linear()(x)) == 0
    results["bf16_cancellation_fixture"] = {
        "materialized_weight_output": a,
        "separate_compact_output": b,
        "scope": "Synthetic rounding witness only; not an explanation of our Qwen failures.",
    }

    pool_namespace = {"activations": {}}
    definitions(trees["1_get_safety_neuron.py"], {"activation_hook"}, pool_namespace)
    values = torch.tensor([[[9.0], [1.0], [2.0]]])
    mask = torch.tensor([[False, True, True]])
    pool_namespace["activation_hook"]("example")(None, None, values)
    upstream_pool = float(pool_namespace["activations"]["example"][0][0, 0])
    masked_pool = float(values.masked_fill(~mask[..., None], -torch.inf).max(1).values[0, 0])
    assert upstream_pool == 9 and masked_pool == 2
    results["padding_fixture"] = {
        "unmasked_max": upstream_pool,
        "valid_token_max": masked_pool,
        "scope": "Controlled activations; no measured change to pretrained safety-neuron sets.",
    }

    # SciPy's finite one-dimensional default z-score uses population std (ddof0).
    # Supply that numeric oracle; no SciPy installation or pickle load is needed.
    weights = np.array([0.001, *([1.0] * 100)])
    z = (weights - weights.mean()) / weights.std(ddof=0)
    selection_namespace = {
        "Dict": Dict,
        "np": np,
        "util": SimpleNamespace(load_dict=lambda _: {"example": weights}),
        "zscore": lambda w: (w - w.mean()) / w.std(ddof=0),
    }
    definitions(trees["2_sn_tune_cluster.py"], {"gather_safety_neurons"}, selection_namespace)
    code_selected = selection_namespace["gather_safety_neurons"]("fixture", 3)["example"].tolist()
    equation_selected = np.where((z > 3) & (weights > 0))[0].tolist()
    assert code_selected == [0] and equation_selected == []
    results["threshold_fixture"] = {
        "code_selected": code_selected,
        "paper_equation_selected": equation_selected,
        "selected_coefficient_z": float(z[0]),
        "scope": "Finite synthetic coefficients; not evidence of frequency in published probes.",
    }

    parser_namespace = {"re": re}
    definitions(trees["util_model.py"], {"extract_label_categories_refusal"}, parser_namespace)
    function = next(
        n for n in trees["2_sn_tune_cluster.py"].body if getattr(n, "name", "") == "compute_asr"
    )
    predicate = next(
        n.test
        for n in ast.walk(function)
        if isinstance(n, ast.If) and ast.unparse(n.test).startswith("'No' in refusal_label")
    )
    counts = {}
    for label in ("Safe", "Unsafe"):
        verdict = f"Safety: {label}\nCategories: None\nRefusal: No"
        parsed = parser_namespace["extract_label_categories_refusal"](verdict)
        assert parsed[0] == label and parsed[2] == "No"
        counts[label] = bool(
            eval(
                compile(ast.Expression(predicate), "<pinned-nest-asr-predicate>", "eval"),
                {"res": verdict, "refusal_label": parsed[2]},
            )
        )
    assert counts == {"Safe": True, "Unsafe": True}
    results["judge_predicate_fixture"] = {
        "counted_as_success": counts,
        "scope": "Controlled judge strings only; published-score impact is unknown.",
    }
    results.update(
        {
            "source_sha256": PINS,
            "torch": torch.__version__,
            "scope": (
                "CPU isolated tensor/parser routines from pinned sources. "
                "No full script, model download, trainer or GPU execution."
            ),
        }
    )
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.source)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
