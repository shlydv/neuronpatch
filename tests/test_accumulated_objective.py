"""Memory-saving separate backwards must implement the same combined objective."""

import copy

import torch

from neuronpatch.losses import preservation_kl


def test_sequential_backward_matches_joint_ce_and_preservation():
    torch.manual_seed(7)
    joint = torch.nn.Linear(3, 5).double()
    staged = copy.deepcopy(joint)
    inputs = torch.randn(2, 4, 3, dtype=torch.float64)
    anchors = torch.randn(2, 3, 3, dtype=torch.float64)
    reference = torch.randn(2, 3, 5, dtype=torch.float64)
    labels = torch.tensor([[1, 2, 3, 4], [0, 3, 1, 2]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    def ce(model):
        return torch.nn.functional.cross_entropy(model(inputs).flatten(0, 1), labels.flatten())

    (ce(joint) + 0.7 * preservation_kl(joint(anchors), reference, mask)).backward()
    ce(staged).backward()
    (0.7 * preservation_kl(staged(anchors), reference, mask)).backward()
    for expected, actual in zip(joint.parameters(), staged.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
