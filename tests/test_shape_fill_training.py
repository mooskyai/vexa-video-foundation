from __future__ import annotations

import torch

from vexa_video.data import StageBSyntheticDataset
from vexa_video.training.trainer import (
    _soft_evaluator_shape_fill,
    _soft_evaluator_shape_fill_loss,
    _soft_evaluator_shape_pair_margin_loss,
)


def _shape_samples() -> tuple[torch.Tensor, torch.Tensor, str, str]:
    dataset = StageBSyntheticDataset(
        length=17,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    )
    square = dataset.sample(0)
    circle = dataset.sample(16)
    return square.video, circle.video, square.caption, circle.caption


def test_soft_evaluator_shape_fill_matches_frozen_geometry() -> None:
    square, circle, _, _ = _shape_samples()
    fill = _soft_evaluator_shape_fill(torch.stack((square, circle)), 32).mean(dim=1)

    assert torch.allclose(fill[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(fill[1], torch.tensor(0.8125), atol=1e-6)


def test_soft_evaluator_shape_fill_loss_penalizes_swapped_labels() -> None:
    square, circle, square_caption, circle_caption = _shape_samples()
    videos = torch.stack((square, circle))

    correct = _soft_evaluator_shape_fill_loss(videos, [square_caption, circle_caption], 32)
    swapped = _soft_evaluator_shape_fill_loss(videos, [circle_caption, square_caption], 32)

    assert float(correct.item()) < 1e-6
    assert float(swapped.item()) > 0.03


def test_soft_evaluator_shape_pair_margin_rejects_collapsed_pair() -> None:
    square, circle, square_caption, circle_caption = _shape_samples()
    separated = _soft_evaluator_shape_pair_margin_loss(
        square.unsqueeze(0),
        [square_caption],
        circle.unsqueeze(0),
        [circle_caption],
        32,
    )
    collapsed = _soft_evaluator_shape_pair_margin_loss(
        square.unsqueeze(0),
        [square_caption],
        square.unsqueeze(0),
        [circle_caption],
        32,
    )

    assert float(separated.item()) < 1e-6
    assert float(collapsed.item()) >= 0.99


def test_soft_evaluator_shape_fill_loss_stays_finite_for_near_empty_foreground() -> None:
    videos = torch.full((2, 3, 8, 32, 32), -0.99, dtype=torch.float32, requires_grad=True)
    captions = [
        "a red square moves right at medium speed",
        "a red circle moves right at medium speed",
    ]

    loss = _soft_evaluator_shape_fill_loss(videos, captions, 32)
    torch.autograd.backward(loss)

    assert torch.isfinite(loss)
    assert videos.grad is not None
    assert torch.isfinite(videos.grad).all()
