"""Check the pinned upstream attribution path without downloading an LLM.

No upstream implementation is vendored. The caller supplies the paper-linked
source; its exact hash is checked before two inspected methods are extracted
and executed in memory on an independent two-feature probability model.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_SHA256 = "2b4e29ccbd3d0ce3a163b54379062e0495b82e288f910125a736d074cf10972c"
COMMIT = "3b02f40eefa8e878db7c5f88849365b25359c5ad"


def extracted_probe(source, *, corrected):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Use the inspected pinned upstream source")
    classes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)]
    generator = next(n for n in classes if n.name == "Generator")
    methods = [
        copy.deepcopy(n)
        for n in generator.body
        if isinstance(n, ast.FunctionDef) and n.name in ("scaled_input", "get_context_attr")
    ]
    assert len(methods) == 2
    gradient_calls = 0
    for method in methods:
        for call in ast.walk(method):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "grad"
            ):
                assert isinstance(call.args[1], ast.Name) and call.args[1].id == "w_ffn_activations"
                gradient_calls += 1
                if corrected:
                    call.args[1] = ast.Name(id="batch_activations", ctx=ast.Load())
    assert gradient_calls == 1
    tree = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ClassDef(name="Probe", bases=[], keywords=[], body=methods, decorator_list=[]),
            ],
            type_ignores=[],
        )
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["Probe"]()


class Tokens:
    def tokenize(self, text):
        return ["answer"]

    def convert_tokens_to_ids(self, token):
        return 0

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[int(text == "with")]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class ProbabilityModel(nn.Module):
    """Valid two-token probabilities with analytically known path derivatives.

    P(answer | x) = .4 + .1 * (1.2*x0 + x1**power), from x=(0,0) to (1,1).
    Exact IG is (.12,.10) for power1 and power3. The latter exposes a possible
    ranking change; both probabilities remain within [0,1] on the whole path.
    """

    def __init__(self, power):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        self.requires_grad_(False)
        self.power = power
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=1),
            layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=self.projection))],
        )

    def forward(self, input_ids, attention_mask):
        features = input_ids[..., None].double().expand(-1, -1, 2).clone()
        hidden = self.projection(features)
        p = 0.4 + 0.1 * (1.2 * hidden[..., 0] + hidden[..., 1] ** self.power)
        return SimpleNamespace(logits=torch.stack([p.log(), (1 - p).log()], -1))


def audit(source):
    rows = []
    points = torch.arange(20, dtype=torch.float64) / 20
    for power in (1, 3):
        values = []
        for corrected in (False, True):
            probe = extracted_probe(source, corrected=corrected)
            probe.model, probe.tokenizer = ProbabilityModel(power), Tokens()
            result = probe.get_context_attr(0, "without", "with", "answer")
            values.append(torch.tensor(result["all_attr_gold"][0], dtype=torch.float64))
        # Closed-form derivative evaluated at the exact same left-endpoint grid.
        derivative = torch.stack(
            [torch.full_like(points, 0.12), 0.1 * power * points ** (power - 1)], -1
        )
        correct = derivative.mean(0)
        endpoint = (points[:, None] * derivative).mean(0)
        torch.testing.assert_close(values[0], endpoint, atol=1e-12, rtol=0)
        torch.testing.assert_close(values[1], correct, atol=1e-12, rtol=0)
        rows.append(
            {
                "power": power,
                "upstream": values[0].tolist(),
                "path_gradient": values[1].tolist(),
                "closed_form_left_riemann": correct.tolist(),
                "exact_integrated_gradients": [0.12, 0.1],
                "endpoint_probability_change": 0.22,
                "upstream_sum": float(values[0].sum()),
                "path_gradient_sum": float(values[1].sum()),
                "upstream_top_feature": int(values[0].argmax()),
                "path_gradient_top_feature": int(values[1].argmax()),
            }
        )
    assert abs(rows[0]["upstream_sum"] / 0.22 - 0.475) < 1e-12
    assert abs(rows[0]["path_gradient_sum"] - 0.22) < 1e-12
    assert rows[1]["upstream_top_feature"] != rows[1]["path_gradient_top_feature"]
    return {
        "status": "Pinned source mismatch reproduced on CPU; real-model effect unmeasured",
        "upstream_commit": COMMIT,
        "source_sha256": SOURCE_SHA256,
        "torch": torch.__version__,
        "dtype": "float64",
        "integration_points": 20,
        "source_execution": (
            "Only inspected scaled_input/get_context_attr methods, extracted in memory"
        ),
        "correction": "Differentiate wrt batch_activations rather than w_ffn_activations",
        "interpretation": (
            "Endpoint differentiation introduces a chain-rule alpha factor. It is not "
            "the paper's stated gradient wrt the interpolated activation. A nonlinear "
            "toy can change rankings. No claim about final selected neurons, reported "
            "benchmark scores, or superiority of neuronpatch follows from this check."
        ),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
