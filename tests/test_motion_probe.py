import torch

from vexa_video.models import MotionProbe


def test_motion_probe_shapes_and_backpropagates() -> None:
    model = MotionProbe()
    video = torch.randn(2, 3, 8, 32, 32)
    direction, color = model(video)
    assert direction.shape == (2, 4)
    assert color.shape == (2, 4)
    (direction.square().mean() + color.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
