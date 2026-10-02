from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset

from .synthetic import (
    COLORS,
    DIRECTIONS,
    SHAPES,
    SyntheticControl,
    SyntheticSample,
    render_motion_sample,
)

_STAGE_B_SPLIT_OFFSETS: dict[str, int] = {
    "train": 0,
    "validation": 10_000_000,
    "test": 20_000_000,
}


def controlled_motion_shape_size(size: int) -> int:
    """Return the shape footprint used by the controlled M1 curriculum."""
    if size < 16:
        raise ValueError("controlled motion size must be >= 16")
    return max(4, size // 4)


def stage_b_control(index: int) -> SyntheticControl:
    """Balanced Stage-B curriculum with fixed medium constant velocity."""
    return SyntheticControl(
        direction=DIRECTIONS[index % len(DIRECTIONS)],
        color=COLORS[(index // len(DIRECTIONS)) % len(COLORS)],
        shape=SHAPES[(index // (len(DIRECTIONS) * len(COLORS))) % len(SHAPES)],
        speed_bucket="medium",
    )


class StageBSyntheticDataset(Dataset[tuple[Tensor, str, Tensor]]):
    def __init__(
        self,
        *,
        length: int,
        frames: int,
        size: int,
        base_seed: int,
        split: str,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if split not in _STAGE_B_SPLIT_OFFSETS:
            raise ValueError(f"split must be one of {tuple(_STAGE_B_SPLIT_OFFSETS)}")
        self.length = length
        self.frames = frames
        self.size = size
        self.base_seed = base_seed
        self.split = split

    def __len__(self) -> int:
        return self.length

    def seed_for_index(self, index: int) -> int:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return self.base_seed + _STAGE_B_SPLIT_OFFSETS[self.split] + index

    def sample(self, index: int) -> SyntheticSample:
        seed = self.seed_for_index(index)
        return render_motion_sample(
            frames=self.frames,
            size=self.size,
            seed=seed,
            control=stage_b_control(index),
            shape_size=controlled_motion_shape_size(self.size),
        )

    def __getitem__(self, index: int) -> tuple[Tensor, str, Tensor]:
        sample = self.sample(index)
        return sample.video, sample.caption, sample.trajectory_xy
