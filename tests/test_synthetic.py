import torch

from vexa_video.data.controlled_motion import (
    StageBSyntheticDataset,
    controlled_motion_shape_size,
)
from vexa_video.data.synthetic import (
    COLORS,
    DIRECTIONS,
    SyntheticControl,
    SyntheticMotionDataset,
    render_motion_sample,
    render_moving_square,
)


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


def test_cardinal_controls_move_in_requested_direction() -> None:
    expected = {
        "right": (1, 0),
        "left": (-1, 0),
        "down": (0, 1),
        "up": (0, -1),
    }
    for direction, signs in expected.items():
        sample = render_motion_sample(
            frames=8,
            size=32,
            seed=11,
            control=SyntheticControl(direction=direction, color="green"),
        )
        delta = sample.trajectory_xy[-1] - sample.trajectory_xy[0]
        assert int(torch.sign(delta[0]).item()) == signs[0]
        assert int(torch.sign(delta[1]).item()) == signs[1]


def test_dataset_balances_direction_and_color_and_separates_split_seeds() -> None:
    train = SyntheticMotionDataset(length=16, frames=8, size=32, base_seed=42, split="train")
    test = SyntheticMotionDataset(length=16, frames=8, size=32, base_seed=42, split="test")
    controls = [train.sample(index).control for index in range(16)]
    assert {control.direction for control in controls} == set(DIRECTIONS)
    assert {control.color for control in controls} == set(COLORS)
    assert {train.seed_for_index(index) for index in range(16)}.isdisjoint(
        {test.seed_for_index(index) for index in range(16)}
    )


def test_controlled_motion_uses_shape_resolving_object_footprint() -> None:
    dataset = StageBSyntheticDataset(
        length=32,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    )
    square = dataset.sample(0)
    circle = dataset.sample(16)

    assert controlled_motion_shape_size(32) == 8
    square_mask = square.video[:, 0].amax(dim=0) > -0.9
    circle_mask = circle.video[:, 0].amax(dim=0) > -0.9
    assert int(square_mask.sum().item()) == 64
    assert int(circle_mask.sum().item()) == 52
