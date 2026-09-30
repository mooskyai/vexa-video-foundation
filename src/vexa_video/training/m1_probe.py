from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from vexa_video.config import ProjectConfig
from vexa_video.data import COLORS, DIRECTIONS, SyntheticMotionDataset
from vexa_video.models import MotionProbe
from vexa_video.training.checkpoint import build_checkpoint


@dataclass(frozen=True, slots=True)
class M1ProbeResult:
    baseline_direction_accuracy: float
    baseline_color_accuracy: float
    direction_accuracy: float
    color_accuracy: float
    gate_passed: bool
    checkpoint: Path


def _batch(
    dataset: SyntheticMotionDataset,
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    samples = [dataset.sample(index) for index in indices]
    videos = torch.stack([sample.video for sample in samples]).to(device)
    direction = torch.tensor(
        [DIRECTIONS.index(sample.control.direction) for sample in samples],
        dtype=torch.long,
        device=device,
    )
    color = torch.tensor(
        [COLORS.index(sample.control.color) for sample in samples],
        dtype=torch.long,
        device=device,
    )
    return videos, direction, color


def _accuracy(
    model: MotionProbe,
    dataset: SyntheticMotionDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    direction_correct = 0
    color_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            videos, direction, color = _batch(dataset, indices, device=device)
            direction_logits, color_logits = model(videos)
            direction_correct += int((direction_logits.argmax(dim=-1) == direction).sum().item())
            color_correct += int((color_logits.argmax(dim=-1) == color).sum().item())
            total += len(indices)
    return direction_correct / total, color_correct / total


def train_m1_probe(
    cfg: ProjectConfig,
    *,
    device: torch.device,
    run_dir: str | Path,
    steps: int | None = None,
) -> M1ProbeResult:
    """Train the M1 supervised sanity probe before generative motion training."""
    if cfg.data.height != cfg.data.width:
        raise ValueError("M1 synthetic probe currently requires square video dimensions")
    target_steps = steps if steps is not None else cfg.m1.probe_steps
    if target_steps <= 0:
        raise ValueError("steps must be positive")

    train_dataset = SyntheticMotionDataset(
        length=cfg.m1.probe_train_samples,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="train",
    )
    eval_dataset = SyntheticMotionDataset(
        length=cfg.m1.probe_eval_samples,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="test",
    )
    model = MotionProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.m1.probe_learning_rate)

    baseline_direction, baseline_color = _accuracy(
        model,
        eval_dataset,
        batch_size=cfg.m1.probe_batch_size,
        device=device,
    )

    generator = torch.Generator().manual_seed(cfg.seed + 1_001)
    log_every = max(1, target_steps // 5)
    for step in range(1, target_steps + 1):
        index_tensor = torch.randint(
            0,
            len(train_dataset),
            (cfg.m1.probe_batch_size,),
            generator=generator,
        )
        indices = [int(value.item()) for value in index_tensor]
        videos, direction, color = _batch(train_dataset, indices, device=device)
        direction_logits, color_logits = model(videos)
        loss = F.cross_entropy(direction_logits, direction) + F.cross_entropy(color_logits, color)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == target_steps:
            print(f"m1_probe step={step} loss={float(loss.detach().item()):.6f}")

    direction_accuracy, color_accuracy = _accuracy(
        model,
        eval_dataset,
        batch_size=cfg.m1.probe_batch_size,
        device=device,
    )
    gate_passed = (
        direction_accuracy >= cfg.m1.probe_direction_gate
        and color_accuracy >= cfg.m1.probe_color_gate
    )

    output_dir = Path(run_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints_dir / "m1-probe.pt"
    torch.save(
        build_checkpoint(
            model=model,
            optimizer=optimizer,
            step=target_steps,
            config=asdict(cfg),
            metadata={
                "milestone": "M1",
                "stage": "supervised-motion-sanity",
                "dataset": "synthetic-cardinal-v1",
                "baseline_direction_accuracy": baseline_direction,
                "baseline_color_accuracy": baseline_color,
                "direction_accuracy": direction_accuracy,
                "color_accuracy": color_accuracy,
                "gate_passed": gate_passed,
            },
        ),
        checkpoint,
    )
    metrics_path = output_dir / "probe-metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "baseline_direction_accuracy": baseline_direction,
                "baseline_color_accuracy": baseline_color,
                "direction_accuracy": direction_accuracy,
                "color_accuracy": color_accuracy,
                "gate_passed": gate_passed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return M1ProbeResult(
        baseline_direction_accuracy=baseline_direction,
        baseline_color_accuracy=baseline_color,
        direction_accuracy=direction_accuracy,
        color_accuracy=color_accuracy,
        gate_passed=gate_passed,
        checkpoint=checkpoint,
    )
