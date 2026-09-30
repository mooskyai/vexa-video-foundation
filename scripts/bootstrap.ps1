$ErrorActionPreference = "Stop"

uv python install 3.14
uv python pin 3.14
uv sync --group dev
uv run vexa-video doctor
uv run pytest
