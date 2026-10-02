from __future__ import annotations

import pytest
import torch

from vexa_video.training.m1_diagnostics import (
    counterfactual_response_metrics,
    linear_cka,
    response_svd_metrics,
)


def test_linear_cka_is_one_for_identical_centered_geometry() -> None:
    samples = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert float(linear_cka(samples, samples).item()) == pytest.approx(1.0, abs=1e-6)


def test_response_svd_detects_rank_one_counterfactuals() -> None:
    coeff = torch.arange(1.0, 9.0).reshape(8, 1)
    direction = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    top1, top4, rank = response_svd_metrics(coeff * direction)
    assert float(top1.item()) == pytest.approx(1.0, abs=1e-6)
    assert float(top4.item()) == pytest.approx(1.0, abs=1e-6)
    assert float(rank.item()) == pytest.approx(1.0, abs=1e-5)


def test_counterfactual_metrics_reward_exact_response() -> None:
    target = torch.tensor([[1.0, -2.0], [-2.0, 1.0], [2.0, 2.0], [-1.0, -1.0]])
    metrics = counterfactual_response_metrics(target, target)
    assert metrics.cosine == pytest.approx(1.0, abs=1e-6)
    assert metrics.magnitude_ratio == pytest.approx(1.0, abs=1e-6)
    assert metrics.linear_cka == pytest.approx(1.0, abs=1e-6)
