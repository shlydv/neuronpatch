"""CPU numerical comparison with the unchanged upstream FiNE selector.

Only Tensor.cuda is redirected to CPU inside this test process. This checks
scoring/ranking on synthetic tensors; it does not test GPU numerical parity,
behavioral effectiveness, mixed-precision training, or editing objectives.
"""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from neuronpatch.contribution import contribution_indices

COMMIT = "a2489cbac0dfe5a956385734065cfa041384e7d1"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    observed = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != COMMIT or subprocess.check_output(
        ["git", "-C", str(args.upstream), "diff", "HEAD"], text=True
    ):
        raise ValueError("Expected unchanged pinned upstream source")
    path = args.upstream / "knowledge-editing/easyeditor/models/fine/utils.py"
    spec = importlib.util.spec_from_file_location("upstream_fine_utils", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    trials = 0
    for seed in range(6):
        torch.manual_seed(seed)
        layers, width, hidden, vocabulary = 5, 16, 8, 23
        modules = [torch.nn.Linear(width, hidden) for _ in range(layers)]
        head = torch.nn.Linear(hidden, vocabulary, bias=False)
        model = SimpleNamespace(
            transformer=SimpleNamespace(
                h=[SimpleNamespace(mlp=SimpleNamespace(fc_out=module)) for module in modules]
            ),
            lm_head=head,
        )
        activations = torch.randn(4, layers, width).half()
        tokens = torch.randint(vocabulary, (4,))
        for excluded in (0, 2, 3):
            hp = SimpleNamespace(last_layer=excluded, model_name="gpt-j-6b")
            with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
                expected = upstream.get_top_neurons(hp, model, activations, tokens, 5)
            actual = contribution_indices(
                activations,
                [m.weight for m in modules],
                head.weight,
                tokens,
                5,
                exclude_last_layers=excluded,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            trials += 1
    print(
        json.dumps(
            {
                "status": "passed",
                "trials": trials,
                "token_positions": trials * 4,
                "upstream_commit": COMMIT,
                "scope": "exact CPU index parity against unchanged upstream scoring routine",
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
