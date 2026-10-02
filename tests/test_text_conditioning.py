from __future__ import annotations

import torch

from vexa_video.models.dit import importance_weighted_text_pool, token_text_attention


def test_importance_weighted_text_pool_prefers_high_energy_projected_token() -> None:
    text = torch.tensor([[[1.0, 0.0], [0.0, 8.0]]])
    mask = torch.tensor([[True, True]])

    pooled = importance_weighted_text_pool(text, mask)

    assert float(pooled[0, 1].item()) > float(pooled[0, 0].item())


def test_projection_driven_text_attention_uses_raw_projected_magnitude() -> None:
    video = torch.tensor([[[0.0, 1.0]]])
    weak = torch.tensor([[[0.0, 0.1], [1.0, 0.0]]])
    strong = torch.tensor([[[0.0, 8.0], [1.0, 0.0]]])
    mask = torch.tensor([[True, True]])

    weak_attended = token_text_attention(video, weak, mask)
    strong_attended = token_text_attention(video, strong, mask)

    assert not torch.allclose(weak_attended, strong_attended)
    assert float(strong_attended[0, 0, 0].item()) < float(weak_attended[0, 0, 0].item())
