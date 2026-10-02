from pathlib import Path

from vexa_video.config import load_config


def test_tiny_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "tiny.toml"
    cfg = load_config(path)
    assert cfg.data.frames == 8
    assert cfg.text.vocab_size == 260
    assert cfg.m1.probe_steps == 250
    assert cfg.m1.probe_direction_gate == 0.95
    assert cfg.vae.spatial_downsample == 2
    assert cfg.vae.temporal_downsample == 2
    assert cfg.m1.vae_semantic_accuracy_gate == 0.95
    assert cfg.m1.vae_warmup_steps == 800
