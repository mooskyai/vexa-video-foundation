from __future__ import annotations

import pytest

from vexa_video.data import COLORS, DIRECTIONS, SHAPES
from vexa_video.training.trainer import _caption_with_control, _caption_with_direction


def test_sampler_caption_controls_enumerate_independent_attributes() -> None:
    caption = "a red square moves right at medium speed"

    directions = [_caption_with_direction(caption, value) for value in DIRECTIONS]
    colors = [_caption_with_control(caption, value, COLORS) for value in COLORS]
    shapes = [_caption_with_control(caption, value, SHAPES) for value in SHAPES]

    assert directions == [
        "a red square moves right at medium speed",
        "a red square moves left at medium speed",
        "a red square moves down at medium speed",
        "a red square moves up at medium speed",
    ]
    assert colors == [
        "a red square moves right at medium speed",
        "a green square moves right at medium speed",
        "a blue square moves right at medium speed",
        "a yellow square moves right at medium speed",
    ]
    assert shapes == [
        "a red square moves right at medium speed",
        "a red circle moves right at medium speed",
    ]


def test_caption_with_control_rejects_unknown_value() -> None:
    caption = "a red square moves right at medium speed"
    with pytest.raises(ValueError, match="unknown control value"):
        _caption_with_control(caption, "purple", COLORS)
