from __future__ import annotations

import torch

from vexa_video.data import StageBSyntheticDataset
from vexa_video.training.trainer import _soft_shape_template_loss


def test_soft_shape_template_loss_prefers_caption_geometry() -> None:
    dataset = StageBSyntheticDataset(
        length=17,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    )
    square = dataset.sample(0)
    circle = dataset.sample(16)
    videos = torch.stack((square.video, circle.video))
    captions = [square.caption, circle.caption]

    correct = _soft_shape_template_loss(videos, captions, 32)
    swapped = _soft_shape_template_loss(videos, [circle.caption, square.caption], 32)

    assert float(correct.item()) < 1e-6
    assert swapped > correct + 0.10


def test_soft_shape_template_loss_stays_finite_for_near_empty_foreground() -> None:
    videos = torch.full((2, 3, 8, 32, 32), -0.99, dtype=torch.float32, requires_grad=True)
    captions = [
        "a red square moves right at medium speed",
        "a red circle moves right at medium speed",
    ]

    loss = _soft_shape_template_loss(videos, captions, 32)
    torch.autograd.backward(loss)

    assert torch.isfinite(loss)
    assert float(loss.item()) < 2.0
    assert videos.grad is not None
    assert torch.isfinite(videos.grad).all()
