import torch

from vexa_video.diffusion.schedule import LinearNoiseSchedule


def test_add_noise_shape_and_determinism() -> None:
    schedule = LinearNoiseSchedule(100, 0.0001, 0.02)
    clean = torch.ones(2, 4, 4, 8, 8)
    noise = torch.zeros_like(clean)
    steps = torch.tensor([0, 99], dtype=torch.long)
    first = schedule.add_noise(clean, noise, steps)
    second = schedule.add_noise(clean, noise, steps)
    assert first.shape == clean.shape
    assert torch.equal(first, second)
    assert first[0].mean() > first[1].mean()
