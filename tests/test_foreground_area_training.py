from __future__ import annotations

import torch

from vexa_video.data import StageBSyntheticDataset
from vexa_video.training.trainer import _soft_foreground_area_loss


def test_soft_foreground_area_loss_penalizes_expansion_and_disappearance() -> None:
    sample = StageBSyntheticDataset(
        length=1,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    ).sample(0)
    exact = sample.video.unsqueeze(0)
    exact_loss = _soft_foreground_area_loss(exact, [sample.caption], 32)

    diffuse = torch.full_like(exact, -1.0)
    diffuse[:, :, :, 4:28, 4:28] = 0.5
    diffuse_loss = _soft_foreground_area_loss(diffuse, [sample.caption], 32)

    disappearing = exact.clone()
    disappearing[:, :, 3] = -1.0
    disappearing_loss = _soft_foreground_area_loss(disappearing, [sample.caption], 32)

    assert float(exact_loss.item()) < 1e-6
    assert diffuse_loss > exact_loss
    assert disappearing_loss > exact_loss
