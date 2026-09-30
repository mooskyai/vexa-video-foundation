import torch

from vexa_video.data.synthetic import render_moving_square


def test_synthetic_sample_is_deterministic() -> None:
    left = render_moving_square(frames=8, size=32, seed=7)
    right = render_moving_square(frames=8, size=32, seed=7)
    assert left.caption == right.caption
    assert torch.equal(left.video, right.video)
    assert torch.equal(left.trajectory_xy, right.trajectory_xy)


def test_trajectory_moves_horizontally() -> None:
    sample = render_moving_square(frames=8, size=32, seed=8)
    x = sample.trajectory_xy[:, 0]
    y = sample.trajectory_xy[:, 1]
    assert torch.all(y == y[0])
    assert x[0] != x[-1]
