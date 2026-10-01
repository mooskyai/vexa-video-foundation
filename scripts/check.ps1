$ErrorActionPreference = "Stop"

uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=vexa_video
