from pathlib import Path

from vexa_video.config import load_config


def test_tiny_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "tiny.toml"
    cfg = load_config(path)
    assert cfg.data.frames == 8
    assert cfg.text.vocab_size == 260
