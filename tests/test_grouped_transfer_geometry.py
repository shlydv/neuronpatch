"""A useful individual-coordinate score need not survive parameter sharing."""

import torch

from neuronpatch.grouped import GroupedRowPatch


def test_individually_helpful_rows_can_hurt_transfer_when_shared():
    # At zero, task gradients are [2, -1], transfer gradients [1, -2].
    # Each coordinate has positive task/transfer alignment. Sharing sums them:
    # task sum=1, transfer sum=-1. The proposed shared direction is harmful.
    x = torch.ones(1, 1, dtype=torch.float64)
    train_target = torch.tensor([[-2.0, 1.0]], dtype=torch.float64)
    transfer_target = torch.tensor([[-1.0, 2.0]], dtype=torch.float64)
    outcomes = []
    for groups in ([0, 1], [0, 0]):
        base = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            base.weight.zero_()
        patch = GroupedRowPatch(base, [0, 1], groups)
        optimizer = torch.optim.Adam([patch.delta], lr=0.01, eps=1e-12)
        train_before = 0.5 * (patch(x) - train_target).square().sum()
        transfer_before = 0.5 * (patch(x) - transfer_target).square().sum()
        train_before.backward()
        optimizer.step()
        with torch.no_grad():
            train_after = 0.5 * (patch(x) - train_target).square().sum()
            transfer_after = 0.5 * (patch(x) - transfer_target).square().sum()
            assert train_after < train_before
            outcomes.append(
                (
                    float(transfer_after - transfer_before),
                    float(patch.merged_linear().weight.norm()),
                )
            )
    assert outcomes[0][0] < 0 < outcomes[1][0]
    # This reversal is not explained by a larger effective-weight step.
    assert abs(outcomes[0][1] - outcomes[1][1]) < 1e-10
