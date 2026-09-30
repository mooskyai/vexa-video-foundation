from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import torch
from torch import nn


def build_checkpoint(
    *,
    model: nn.Module,
    step: int,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "step": step,
        "model": model.state_dict(),
        "config": config,
        "metadata": metadata or {},
        "torch_rng_state": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload
