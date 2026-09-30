from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

DIRECTIONS: tuple[str, ...] = ("right", "left", "down", "up")
COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow")
SHAPES: tuple[str, ...] = ("square", "circle")
SPEED_BUCKETS: tuple[str, ...] = ("slow", "medium", "fast")

_COLOR_VALUES: dict[str, tuple[float, float, float]] = {
    "red": (1.0, -0.65, -0.65),
    "green": (-0.65, 1.0, -0.65),
    "blue": (-0.65, -0.65, 1.0),
    "yellow": (1.0, 1.0, -0.65),
}
_SPEED_PIXELS: dict[str, int] = {"slow": 1, "medium": 2, "fast": 3}
_SPLIT_OFFSETS: dict[str, int] = {
    "train": 0,
    "validation": 10_000_000,
    "test": 20_000_000,
}


@dataclass(frozen=True, slots=True)
class SyntheticControl:
    direction: str
    color: str
    shape: str = "square"
    speed_bucket: str = "medium"

    def validate(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction: {self.direction}")
        if self.color not in COLORS:
            raise ValueError(f"unknown color: {self.color}")
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape: {self.shape}")
        if self.speed_bucket not in SPEED_BUCKETS:
            raise ValueError(f"unknown speed bucket: {self.speed_bucket}")


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    video: Tensor
    caption: str
    trajectory_xy: Tensor
    control: SyntheticControl
    seed: int


def _balanced_control(index: int) -> SyntheticControl:
    return SyntheticControl(
        direction=DIRECTIONS[index % len(DIRECTIONS)],
        color=COLORS[(index // len(DIRECTIONS)) % len(COLORS)],
        speed_bucket=SPEED_BUCKETS[(index // (len(DIRECTIONS) * len(COLORS))) % len(SPEED_BUCKETS)],
        shape=SHAPES[(index // (len(DIRECTIONS) * len(COLORS) * len(SPEED_BUCKETS))) % len(SHAPES)],
    )


def _paint_shape(
    video: Tensor,
    *,
    frame: int,
    x: int,
    y: int,
    shape_size: int,
    control: SyntheticControl,
) -> None:
    color = torch.tensor(_COLOR_VALUES[control.color], dtype=video.dtype, device=video.device)
    patch = video[:, frame, y : y + shape_size, x : x + shape_size]
    if control.shape == "square":
        patch.copy_(color[:, None, None].expand_as(patch))
        return

    coords = torch.arange(shape_size, dtype=video.dtype, device=video.device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (shape_size - 1) / 2.0
    radius = max(1.0, shape_size / 2.0)
    mask = (xx - center).square() + (yy - center).square() <= radius**2
    for channel, value in enumerate(color):
        patch[channel][mask] = value


def render_motion_sample(
    *,
    frames: int,
    size: int,
    seed: int,
    control: SyntheticControl,
    shape_size: int | None = None,
) -> SyntheticSample:
    if frames < 2 or size < 16:
        raise ValueError("frames must be >= 2 and size must be >= 16")
    control.validate()
    shape_pixels = shape_size or max(3, size // 8)
    if shape_pixels >= size:
        raise ValueError("shape_size must be smaller than size")

    max_step = max(1, (size - shape_pixels) // (frames - 1))
    step = min(_SPEED_PIXELS[control.speed_bucket], max_step)
    dx, dy = {
        "right": (step, 0),
        "left": (-step, 0),
        "down": (0, step),
        "up": (0, -step),
    }[control.direction]
    total_dx = dx * (frames - 1)
    total_dy = dy * (frames - 1)
    max_xy = size - shape_pixels

    min_x = max(0, -total_dx)
    max_x = min(max_xy, max_xy - total_dx)
    min_y = max(0, -total_dy)
    max_y = min(max_xy, max_xy - total_dy)
    if min_x > max_x or min_y > max_y:
        raise ValueError("video geometry cannot contain the requested motion")

    generator = torch.Generator().manual_seed(seed)
    start_x = int(torch.randint(min_x, max_x + 1, (1,), generator=generator).item())
    start_y = int(torch.randint(min_y, max_y + 1, (1,), generator=generator).item())

    video = torch.full((3, frames, size, size), -1.0, dtype=torch.float32)
    trajectory = torch.zeros((frames, 2), dtype=torch.float32)
    for frame in range(frames):
        x = start_x + dx * frame
        y = start_y + dy * frame
        _paint_shape(
            video,
            frame=frame,
            x=x,
            y=y,
            shape_size=shape_pixels,
            control=control,
        )
        trajectory[frame] = torch.tensor([x, y], dtype=torch.float32)

    caption = (
        f"a {control.color} {control.shape} moves {control.direction} "
        f"at {control.speed_bucket} speed"
    )
    return SyntheticSample(
        video=video,
        caption=caption,
        trajectory_xy=trajectory,
        control=control,
        seed=seed,
    )


def render_moving_square(
    *,
    frames: int,
    size: int,
    seed: int,
    square_size: int | None = None,
) -> SyntheticSample:
    """Backward-compatible M0 renderer used by the starter smoke path."""
    generator = torch.Generator().manual_seed(seed)
    direction = "right" if bool(torch.randint(0, 2, (1,), generator=generator).item()) else "left"
    return render_motion_sample(
        frames=frames,
        size=size,
        seed=seed,
        shape_size=square_size,
        control=SyntheticControl(
            direction=direction,
            color="red",
            shape="square",
            speed_bucket="medium",
        ),
    )


class SyntheticMotionDataset(Dataset[tuple[Tensor, str, Tensor]]):
    def __init__(
        self,
        length: int,
        frames: int,
        size: int,
        base_seed: int = 0,
        split: str = "train",
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if split not in _SPLIT_OFFSETS:
            raise ValueError(f"split must be one of {tuple(_SPLIT_OFFSETS)}")
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
        return self.base_seed + _SPLIT_OFFSETS[self.split] + index

    def sample(self, index: int) -> SyntheticSample:
        seed = self.seed_for_index(index)
        return render_motion_sample(
            frames=self.frames,
            size=self.size,
            seed=seed,
            control=_balanced_control(index),
        )

    def __getitem__(self, index: int) -> tuple[Tensor, str, Tensor]:
        sample = self.sample(index)
        return sample.video, sample.caption, sample.trajectory_xy
