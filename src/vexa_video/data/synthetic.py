from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    video: Tensor
    caption: str
    trajectory_xy: Tensor


def render_moving_square(
    *,
    frames: int,
    size: int,
    seed: int,
    square_size: int | None = None,
) -> SyntheticSample:
    if frames < 2 or size < 16:
        raise ValueError("frames must be >= 2 and size must be >= 16")
    g = torch.Generator().manual_seed(seed)
    square = square_size or max(3, size // 8)
    max_xy = size - square
    start_x = int(torch.randint(0, max(1, max_xy // 3), (1,), generator=g).item())
    start_y = int(torch.randint(0, max(1, max_xy), (1,), generator=g).item())
    direction = 1 if bool(torch.randint(0, 2, (1,), generator=g).item()) else -1
    if direction < 0:
        start_x = max_xy - start_x

    video = torch.full((3, frames, size, size), -1.0, dtype=torch.float32)
    trajectory = torch.zeros((frames, 2), dtype=torch.float32)

    for frame in range(frames):
        progress = frame / (frames - 1)
        if direction > 0:
            x = round(start_x + progress * (max_xy - start_x))
        else:
            x = round(start_x - progress * start_x)
        y = start_y
        video[0, frame, y : y + square, x : x + square] = 1.0
        video[1, frame, y : y + square, x : x + square] = -0.25
        video[2, frame, y : y + square, x : x + square] = -0.25
        trajectory[frame] = torch.tensor([x, y], dtype=torch.float32)

    word = "right" if direction > 0 else "left"
    return SyntheticSample(
        video=video,
        caption=f"a red square moves {word}",
        trajectory_xy=trajectory,
    )


class SyntheticMotionDataset(Dataset[tuple[Tensor, str, Tensor]]):
    def __init__(self, length: int, frames: int, size: int, base_seed: int = 0) -> None:
        self.length = length
        self.frames = frames
        self.size = size
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, str, Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        sample = render_moving_square(
            frames=self.frames,
            size=self.size,
            seed=self.base_seed + index,
        )
        return sample.video, sample.caption, sample.trajectory_xy
