"""CPU correctness demo: recover a known neuron edit on unseen inputs.

This is a synthetic regression problem, not an LLM behavior benchmark.
Run: python examples/fit_toy.py
"""

from copy import deepcopy
from tempfile import TemporaryDirectory

import torch
from torch import nn

from neuronpatch import NeuronPatch


def main():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 4))
    original = deepcopy(model)
    teacher = deepcopy(model)
    with torch.no_grad():
        teacher[2].weight[:, 3] += torch.tensor([0.5, -0.3, 0.2, 0.4])
    train_x, test_x = torch.randn(256, 6), torch.randn(256, 6)
    with torch.no_grad():
        train_y, test_y = teacher(train_x), teacher(test_x)
        initial = nn.functional.mse_loss(model(test_x), test_y).item()
    patch = NeuronPatch(model, {"2": [3]})
    optimizer = torch.optim.Adam(patch.trainable_parameters(), lr=0.03)
    for _ in range(200):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(train_x), train_y).backward()
        optimizer.step()
    with torch.no_grad():
        final = nn.functional.mse_loss(model(test_x), test_y).item()
        merged = patch.merged_model()
        torch.testing.assert_close(model(test_x), merged(test_x))
        with TemporaryDirectory() as directory:
            patch.save(f"{directory}/patch", metadata={"task": "synthetic known-column edit"})
            reloaded = NeuronPatch.load(original, f"{directory}/patch")
            torch.testing.assert_close(model(test_x), reloaded.model(test_x))
    print(f"Trainable parameters: {patch.num_trainable_parameters}")
    print(f"Held-out MSE: {initial:.6g} -> {final:.6g}")
    print("Save/load and merged-output checks passed.")
    if final >= initial * 0.001:
        raise RuntimeError("Known-edit recovery did not converge")


if __name__ == "__main__":
    main()
